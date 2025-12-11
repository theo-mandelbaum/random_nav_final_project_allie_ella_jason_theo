from enum import IntEnum

class States(IntEnum):
    STARTUP = 0
    DECIDING = 1
    TRAVELING = 2
    READY = 3
    ORANGE_SPOTTED = 4
    RETURNING = 6
    VICTIM_INTERACTION = 7
    ALIGNING_FOR_PHOTO = 8


"""Demonstration of using the nav2 action interface in Python.


This node navigates to a goal pose provided on the command line. This
code also include a demonstration of interacting with OccupancyGrid
messages through the map_util.Map class.


DO NOT MODIFY OR IMPORT THIS FILE. It is only provided as an
illustration.


Author: Nathan Sprague and Kevin Molloy
Version: 10/24/2023


"""
import argparse
import time
import numpy as np
import math
import random
import asyncio
import os
from datetime import datetime

from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from std_msgs.msg import Empty

import rclpy
import rclpy.node
from rclpy.action.client import ActionClient
from rclpy.task import Future
from action_msgs.msg import GoalStatus

from nav2_msgs.action import NavigateToPose

from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from queue import PriorityQueue

from nav_msgs.msg import OccupancyGrid
from jmu_ros2_util import map_utils
import tf_transformations
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from geometry_msgs.msg import TwistStamped

from geometry_msgs.msg import PointStamped, PoseStamped
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs

from rclpy.node import Node
from ros2_aruco_interfaces.msg import ArucoMarkers
from zeta_competition_interfaces.msg import Victim
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError

class RandomNavNode(rclpy.node.Node):

    MISSION_DURATION = 300.00
    
    def __init__(self):

        super().__init__('rand_nav')

        # Create dictionary to track unvisited victims
        self.victims = {}

        # Listen for aruco marker topics
        self.create_subscription(ArucoMarkers, '/aruco_markers', self.marker_callback, 10)

        # Publish our victim node
        self.victim_pub = self.create_publisher(Victim, '/victim', 10)

        # Publisher for victim images (for rqt_image_view etc)
        self.victim_image_pub = self.create_publisher(Image, '/victim_image', 10)

        # QoS for map
        latching_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        # Set up the occupancy grid
        self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            qos_profile=latching_qos
        )

        # Set up the action client
        self.ac = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.get_logger().info("Waiting for NavigateToPose action server...")
        self.ac.wait_for_server()
        self.get_logger().info("NavigateToPose action server is up.")

        # Variables
        self.current_goal = None
        self.start_pose = None
        self.current_pose = None
        self.mission_start = None

        # Return logic 
        self.return_goal = None
        self.return_goal_sent = False

        # Timer that periodically checks remaining time to return
        self.return_timer = self.create_timer(2.5, self.timer_callback)

        # Create buffer and transform it
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)

        # completion future
        self.future_event = Future()

        self.victim_capture_in_progress = False

        self.map = None
        self.goal_timer = None
        self.current_goal_handle = None
        self.goal_timeout_seconds = 30.0

        self.cost_time = None
        self.state = States.STARTUP

        # Publisher for manual spin
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.spin_done = False
        self.spin_timer = None

        self.amcl_ready = False
        self.map_ready = False
        self.bridge = CvBridge()
        
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_callback,
            10
        )

        self.latest_image = None
        self.pending_aruco_id = None
        self.pending_aruco_pose = None

        self.perception_timer = self.create_timer(0.5, self.mark_visible_blocks_visited)

        # Publisher for debugging blocks in Rviz
        self.debug_pub = self.create_publisher(MarkerArray, '/debug_grid_blocks', 10)

        # Subscribe to the camera image
        self.create_subscription(Image, '/oakd/rgb/preview/image_raw', self.camera_callback, 10)

        # Debug Publisher: "What does the robot think is orange?"
        self.debug_img_pub = self.create_publisher(Image, '/camera/debug_mask', 10)

        # Debug Publisher: "Where are the confirmed victims?"
        self.victim_marker_pub = self.create_publisher(MarkerArray, '/debug_victims', 10)

        # Subscribe for competition report requests
        self.create_subscription(Empty, '/report_requested', self.report_requested_callback, 10)

        self.goal_seq = 0

        self.photo_align_timer = None
        self.photo_align_started = False



    # =========================================================================
    #                               STATE TRANSITIONS
    # =========================================================================
    def set_state(self, new_state):
        """Centralized state transition handler."""
        # Rule: Once we are RETURNING, we cannot go back to exploring or victim hunting
        if self.state == States.RETURNING and new_state != States.RETURNING:
            self.get_logger().warn(f"Ignored transition to {new_state} because we are RETURNING.")
            return False

        self.get_logger().info(f"Transition: {self.state} -> {new_state}")
        self.state = new_state
        return True
     
    # =========================================================================
    #                               CORE CALLBACKS
    # =========================================================================

    def amcl_callback(self, msg: PoseWithCovarianceStamped):
        self.current_pose = msg.pose.pose

        if self.start_pose is None:
            self.start_pose = self.current_pose
            self.get_logger().info("Start pose recorded from AMCL.")

        if not self.amcl_ready:
            self.amcl_ready = True
            self.get_logger().info("AMCL pose is now ready.")

        if self.state == States.STARTUP:
            # If everything is ready and we have not sent the first goal yet
            if self.map_ready and self.spin_done:
                self.get_logger().info("STARTUP finished. Transitioning to DECIDING.")

                self.set_state(States.DECIDING)

                if self.mission_start is None:
                    self.mission_start = self.get_clock().now()
                # Send the first goal
                self.send_goal()

    def map_callback(self, map_msg):
        self.get_logger().info("Map updated.")
        if self.map is None:
            self.map = map_utils.Map(map_msg)
            self.map_ready = True
            self.get_logger().info("Map received.")

            # Start a spin to help AMCL localize
            if not self.spin_done:
                self.start_spin()

        self.get_logger().info("Map received! Pre-computing safe zones...")
        self.precompute_static_blocks(map_msg)
        self.get_logger().info(f"Map processing complete. Found {len(self.static_blocks)} valid blocks.")
    
    def timer_callback(self):
        if self.state == States.STARTUP or self.mission_start is None:
            return

        now = self.get_clock().now()
        elapsed = (now - self.mission_start).nanoseconds / 1e9
        remaining = self.MISSION_DURATION - elapsed

        if remaining <= 0:
            self.get_logger().warn("Mission time exceeded.")
            if not self.future_event.done():
                self.future_event.set_result(False)
            return

        t = estimate_return_time(self.start_pose, self.current_pose)
        if t is None:
            return

        safety_buffer = 20.0

        if remaining < t + safety_buffer and not self.state == States.RETURNING:
            if self.start_pose is None:
                self.get_logger().warn("Time to return, but start_pose unknown.")
                return

            # Create the return goal and mark that we are switching to return mode
            start_x = self.start_pose.position.x
            start_y = self.start_pose.position.y
            self.return_goal = create_nav_goal_at(start_x, start_y, 0.0)
            self.set_state(States.RETURNING)
            self.get_logger().info("Time to go home. Queued return goal and canceling current goal.")

            # Cancel the current goal so that nav2 stops pursuing it
            if self.current_goal_handle is not None:
                # Stop per-goal timers so they do not interfere
                if self.goal_timer is not None:
                    self.goal_timer.cancel()
                    self.goal_timer = None
                self.cost_time = None

                cancel_future = self.current_goal_handle.cancel_goal_async()
                cancel_future.add_done_callback(self.goal_cancelled_callback)
            else:
                # No current goal, send return goal immediately
                self.get_logger().info("No current goal, sending return goal now.")
                self.send_goal()

    def camera_callback(self, msg):
        self.latest_image = msg
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError:
            return

        # 1. Image Processing
        img_int = cv_image.astype(np.int32)
        ideal_orange = np.array([0, 165, 255])  # BGR
        diff = img_int - ideal_orange 
        diff_sq = diff ** 2
        distance_map = np.sum(diff_sq, axis=2)
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(distance_map.astype(np.float32))
        dist_score = np.sqrt(min_val)

        # 2. VISUAL DEBUGGING: Create a Black/White mask
        mask = np.zeros_like(distance_map, dtype=np.uint8)
        mask[np.sqrt(distance_map) < 100] = 255 
        
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(mask, "mono8")
            self.debug_img_pub.publish(debug_msg)
        except CvBridgeError:
            pass

        # 3. Detection Logic
        pass
        # if dist_score > 100:
        #     return  # Not orange enough

        # valid_states = [States.TRAVELING, States.DECIDING]
        
        # if self.state in valid_states:
        #     # LOOP PREVENTION: 
        #     # Check if we are near a known victim (use 1.5m radius for camera)
        #     if self.current_pose and not self.within_two(self.current_pose, threshold=1.5):
        #         return 

        #     self.get_logger().warn("ORANGE DETECTED! Stopping.")
        #     self.stop_navigation()
        #     self.set_state(States.ORANGE_SPOTTED)
        #     self.handle_orange_event()

    # =========================================================================
    #                               GOAL LOGIC
    # =========================================================================

    def stop_navigation(self):
        if self.current_goal_handle is not None:
            self.get_logger().info("Cancelling current navigation goal...")
            future = self.current_goal_handle.cancel_goal_async()
            self.current_goal_handle = None
        
        stop_msg = TwistStamped()
        stop_msg.header.stamp = self.get_clock().now().to_msg()
        self.cmd_pub.publish(stop_msg)

    def publish_victim_markers(self):
        markers = MarkerArray()
        del_msg = Marker()
        del_msg.action = Marker.DELETEALL
        del_msg.ns = "victims"
        markers.markers.append(del_msg)

        for i, (x, y, z, vid) in enumerate(self.victims.keys()):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "victims"
            m.id = i + 1000 
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.5
            m.scale.x = 0.3
            m.scale.y = 0.3
            m.scale.z = 0.3
            m.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)
            markers.markers.append(m)
            
        self.victim_marker_pub.publish(markers)

    def handle_orange_event(self):
        if self.current_pose is None:
            return

        # Robot pose in map
        rx = self.current_pose.position.x
        ry = self.current_pose.position.y
        rz = self.current_pose.position.z

        # Get robot yaw from quaternion
        q = self.current_pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

        # Assume the orange block is about 0.7 m in front of the robot
        victim_dist = 0.7
        vx = rx + victim_dist * math.cos(yaw)
        vy = ry + victim_dist * math.sin(yaw)
        vz = rz

        orange_id = -random.randint(1, 1000)

        self.get_logger().info(f"New Orange Victim recorded at ({vx:.2f}, {vy:.2f})")
        self.victims[(vx, vy, vz, orange_id)] = False

        self.publish_victim_markers()
        self.set_state(States.VICTIM_INTERACTION)

        self.resume_timer = self.create_timer(4.0, self.resume_mission_callback)

    def resume_mission_callback(self):
        self.resume_timer.cancel()
        self.resume_timer = None
        self.get_logger().info("Interaction complete. Resuming mission.")
        
        if self.state == States.RETURNING or self.state == States.STARTUP:
            return

        self.set_state(States.DECIDING)
        self.send_goal()

    def send_goal(self):
        if self.state == States.RETURNING:
            if self.return_goal_sent:
                self.get_logger().info("Return goal already sent. Ignoring request.")
                return self.future_event
            
            next_goal = self.return_goal
            self.return_goal_sent = True
            self.get_logger().info("Sending return to start goal.")
            
        else:
            # 1. Calculate Mission Progress (0.0 to 1.0)
            now = self.get_clock().now()
            elapsed = (now - self.mission_start).nanoseconds / 1e9
            progress = min(elapsed / self.MISSION_DURATION, 1.0)

            # 2. Get Best Goals
            if self.current_pose is None or self.start_pose is None:
                self.get_logger().warn("Poses not ready, waiting...")
                return

            queue = self.get_best_goals(
                self.current_pose.position.x, 
                self.current_pose.position.y, 
                self.start_pose.position.x, 
                self.start_pose.position.y, 
                progress
            )

            # 3. Handle Empty Queue
            if queue.empty():
                self.get_logger().warn("No valid goals found! Using random fallback.")
                next_goal = create_random_free_goal(self.map)
            else:
                # Choose from top 3 scoring blocks
                num_candidates = min(3, queue.qsize())
                candidates = [queue.get() for _ in range(num_candidates)]
                score, best_block = random.choice(candidates)
            
                self.get_logger().info(
                    f"Selected block among top {num_candidates} candidates (score={score})"
                )
                next_goal = self.generate_random_point(best_block)
                
                self.publish_debug_grid()
            if self.current_pose:
                self.goal_timeout_seconds = estimate_return_time(next_goal.pose.pose, self.current_pose) + 10.0
                self.get_logger().info(f"Dynamic Timeout set to: {self.goal_timeout_seconds:.2f}s")

        self.current_goal = next_goal
        next_goal.pose.header.stamp = self.get_clock().now().to_msg()
        pos = next_goal.pose.pose.position

        self.get_logger().info(
            f"Sending new goal to navigation server at ({pos.x:.2f}, {pos.y:.2f})"
        )
        self.goal_seq += 1
        goal_seq = self.goal_seq

        self.goal_future = self.ac.send_goal_async(next_goal)
        self.goal_future.add_done_callback(
            lambda future, seq=goal_seq: self.goal_response_callback(future, seq)
        )


        return self.future_event

    def goal_response_callback(self, future, seq):
        # Ignore responses from old goals
        if seq != self.goal_seq:
            self.get_logger().info(
                f"Ignoring response for old goal seq {seq}, current seq {self.goal_seq}"
            )
            return
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected, sending next.")
            self.send_goal()
            return

        self.get_logger().info("Goal accepted!")
        self.current_goal_handle = goal_handle

        if self.state == States.DECIDING:
            self.set_state(States.TRAVELING)

        if self.goal_timer is not None:
            self.goal_timer.cancel()
            self.goal_timer = None

        if self.state not in [States.RETURNING]:
            self.goal_timer = self.create_timer(
                self.goal_timeout_seconds,
                self.goal_timeout_callback
            )

        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(
            lambda future, seq=seq: self.goal_result_callback(future, seq)
        )

    def goal_result_callback(self, future, seq):
        # Ignore results from old, superseded goals
        if seq != self.goal_seq:
            self.get_logger().info(
                f"Ignoring result for old goal seq {seq}, current seq {self.goal_seq}"
            )
            return
        result = future.result()
        status = result.status

        # Stop timeout timer
        if self.goal_timer is not None:
            self.goal_timer.cancel()
            self.goal_timer = None

        self.cost_time = None

        self.get_logger().info(f"Goal Ended: {status}. Current State: {self.state}")
        
        if status == GoalStatus.STATUS_SUCCEEDED:
            if self.state == States.RETURNING:
                self.get_logger().info("Return goal succeeded, shutting down.")
                if not self.future_event.done():
                    self.future_event.set_result(True)
                # Don't send any more goals
                self.current_goal_handle = None
                return

            elif self.state == States.ALIGNING_FOR_PHOTO:
                self.get_logger().info("Final alignment goal succeeded. Capturing victim.")
                if not self.victim_capture_in_progress:
                    self.victim_capture_in_progress = True
                    self.finalize_victim_capture()
                self.current_goal_handle = None
                return

            else:
                self.get_logger().info("Goal succeeded.")
                if not self.state == States.RETURNING:
                    self.send_goal()

        elif status in [GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_CANCELED]:

            if self.state == States.ALIGNING_FOR_PHOTO:
                self.get_logger().warn("Alignment goal failed; capturing anyway.")
                if not self.victim_capture_in_progress:
                    self.victim_capture_in_progress = True
                    self.finalize_victim_capture()
                return

            # # Case 1: approach to an ArUco victim failed
            # if self.state == States.TAKING_PHOTO and self.pending_aruco_id is not None:
            #     self.get_logger().warn(
            #         f"Approach to ArUco {self.pending_aruco_id} failed with status {status}. "
            #         f"Skipping this victim and resuming mission."
            #     )
            #     # Drop this victim and go back to exploring
            #     self.pending_aruco_id = None
            #     self.pending_aruco_pose = None
            #     self.set_state(States.DECIDING)
            #     self.send_goal()
            #     return

            # Case 2: orange victim flow, where cancelling a goal is intentional
            if self.state in [States.ORANGE_SPOTTED, States.VICTIM_INTERACTION]:
                self.get_logger().info("Goal cancellation was intentional for victim. Waiting for timer...")
                return

            # Existing RETURNING / other fail logic
            self.get_logger().warn(f"Goal failed/canceled (Status: {status})")

            if self.state == States.RETURNING and self.return_goal_sent:
                self.get_logger().warn("Return goal failed. Mission ending.")
                if not self.future_event.done():
                    self.future_event.set_result(True)

            elif self.state == States.RETURNING and not self.return_goal_sent:
                self.get_logger().info("Previous goal interrupted. Sending return goal now.")
                self.send_goal()

            else:
                self.set_state(States.DECIDING)
                self.send_goal()


        self.current_goal_handle = None

    def goal_timeout_callback(self):
        if self.state == States.RETURNING:
            return

        self.get_logger().warn("TIMEOUT: Goal exceeded timeout, cancelling...")

        # Stop the timer so it does not fire again
        if self.goal_timer is not None:
            self.goal_timer.cancel()
            self.goal_timer = None

        # Cancel the goal if still active
        if self.current_goal_handle is not None:
            cancel_future = self.current_goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self.goal_cancelled_callback)

    def goal_cancelled_callback(self, future):
        # Clear the current goal handle immediately
        self.current_goal_handle = None

    # =========================================================================
    #                         REPORT REQUESTED CALLBACK
    # =========================================================================
    def report_requested_callback(self, msg):
        """Handle competition report request by publishing all known victims."""
        self.get_logger().info("Report requested. Publishing all known victims.")
        for key, data in self.victims.items():
            v = Victim()
            v.id = key[3]
            v.point.header.frame_id = "map"
            v.point.point = data["pose"].position
            v.image = data["image"]
            self.victim_pub.publish(v)
        self.get_logger().info("Finished publishing victim reports.")


    # =========================================================================
    #                              HELPERS
    # =========================================================================
    
    def start_spin(self):
        """Do a short spin in place to help AMCL get observations."""
        if self.spin_timer is not None or self.spin_done:
            return

        self.get_logger().info("Starting AMCL localization spin.")

        # Creates a twist stamped object which tells the robot to turn 0.8 radians per second
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.twist.angular.z = 0.8  # rad/s
        start_time = self.get_clock().now()

        def spin_step():
            if self.state != States.STARTUP:
                return
            # Gets the elapsed time from the start time
            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed > 4.0:
                # stop
                stop_msg = TwistStamped()
                stop_msg.header.stamp = self.get_clock().now().to_msg()
                self.cmd_pub.publish(stop_msg)
                self.get_logger().info("Finished localization spin.")
                self.spin_timer.cancel()
                self.spin_timer = None
                self.spin_done = True

            twist.header.stamp = self.get_clock().now().to_msg()
            self.cmd_pub.publish(twist)

        # Calls spin_step every 0.05 seconds
        self.spin_timer = self.create_timer(0.05, spin_step)
    
    def precompute_static_blocks(self, msg):
        self.static_blocks = []
        
        # 1. Setup
        grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        resolution = msg.info.resolution
        ox, oy = msg.info.origin.position.x, msg.info.origin.position.y
        
        block_size = 10 
        
        self.grid_height, self.grid_width = grid.shape
        self.n_rows = self.grid_height // block_size
        self.n_cols = self.grid_width // block_size
        
        # Store scores per block
        self.score_grid = np.zeros((self.n_rows, self.n_cols))

        # PASS 1: Calculate raw scores
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                gy = r * block_size
                gx = c * block_size
                gy_end = min(gy + block_size, self.grid_height)
                gx_end = min(gx + block_size, self.grid_width)
                
                block = grid[gy:gy_end, gx:gx_end]
                num_free = np.sum(block == 0)
                num_occ = np.sum(block > 50)
                
                if num_free == 0:
                    self.score_grid[r, c] = -9999.0
                else:
                    self.score_grid[r, c] = float(num_free) - (5.0 * float(num_occ))

        # Generate block list
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.score_grid[r, c] < -1000:
                    continue

                gy = r * block_size
                gx = c * block_size
                center_x = ox + (gx + block_size / 2) * resolution
                center_y = oy + (gy + block_size / 2) * resolution

                self.static_blocks.append({
                    'gx': gx, 'gy': gy,
                    'r': r, 'c': c,
                    'cx': center_x, 'cy': center_y,
                    'size': block_size,
                    'base_score': self.score_grid[r, c]
                })

    def get_dynamic_smoothed_score(self, r, c):
        center_score = self.score_grid[r, c]
        
        if center_score < -1000:
            return center_score

        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbor_sum = 0.0
        neighbor_count = 0

        for (dr, dc) in deltas:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols:
                n_score = self.score_grid[nr, nc]
                neighbor_sum += n_score
                neighbor_count += 1
        
        if neighbor_count > 0:
            avg_neighbor = neighbor_sum / neighbor_count
            return (0.6 * center_score) + (0.4 * avg_neighbor)
        else:
            return center_score

    def get_best_goals(self, robot_x, robot_y, start_x, start_y, progress):
        if not hasattr(self, 'static_blocks') or not self.static_blocks:
            return PriorityQueue()

        q = PriorityQueue()

        # Weights
        w_free = 1.0
        w_frontier = 250.0
        w_robot_dist = 0.01
        w_start_dist = 0.002
        w_visited = 150.0

        visited_threshold = -300  # If score_cell < this, treat as visited

        for block in self.static_blocks:
            r = block['r']
            c = block['c']
            base = self.score_grid[r, c]

            if base < -900:
                continue

            free_score = max(base, 0)

            frontier_score = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols:
                    if self.score_grid[nr, nc] > visited_threshold:
                        frontier_score += 1

            dist_robot = math.hypot(block['cx'] - robot_x, block['cy'] - robot_y)
            dist_start = math.hypot(block['cx'] - start_x, block['cy'] - start_y)

            visited_penalty = w_visited if base < visited_threshold else 0
            start_weight = (4 * progress - 2) * w_start_dist

            final_score = (
                + w_free * free_score
                + w_frontier * frontier_score
                - w_robot_dist * dist_robot
                - start_weight * dist_start
                - visited_penalty
            )

            block['dynamic_score'] = final_score
            q.put((-final_score, block))

        return q

    def mark_visible_blocks_visited(self):
        """
        Marks blocks as visited if they are within the camera FOV.
        """
        if self.current_pose is None or not hasattr(self, 'static_blocks'):
            return

        max_scan_range = 1.5
        fov_rad = math.radians(50)

        rx = self.current_pose.position.x
        ry = self.current_pose.position.y
        
        q = self.current_pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

        for block in self.static_blocks:
            if block['base_score'] < -500:
                continue

            dx = block['cx'] - rx
            dy = block['cy'] - ry
            dist = math.hypot(dx, dy)

            if dist > max_scan_range:
                continue

            angle_to_block = math.atan2(dy, dx)
            angle_diff = math.atan2(math.sin(angle_to_block - yaw), math.cos(angle_to_block - yaw))

            if abs(angle_diff) < (fov_rad / 2.0):
                self.score_grid[block['r'], block['c']] -= 500
                block['base_score'] -= 500

    def get_heatmap_color(self, value):
        c = ColorRGBA()
        c.a = 0.5

        value = max(0.0, min(1.0, value))

        if value < 0.5:
            c.r = 1.0
            c.g = 2.0 * value
            c.b = 0.0
        else:
            c.r = 2.0 * (1.0 - value)
            c.g = 1.0
            c.b = 0.0
        return c
    
    def publish_debug_grid(self, best_block=None):
        if not hasattr(self, 'static_blocks') or not self.static_blocks:
            return

        markers = MarkerArray()
        
        delete_msg = Marker()
        delete_msg.action = Marker.DELETEALL
        markers.markers.append(delete_msg)

        current_time = self.get_clock().now().to_msg()
        res = self.map.resolution

        max_score = 400.0
        min_score = -400.0
        score_range = max_score - min_score
        
        for i, block in enumerate(self.static_blocks):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = current_time
            marker.ns = "grid_blocks"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            marker.pose.position.x = block['cx']
            marker.pose.position.y = block['cy']
            marker.pose.position.z = 0.05
            
            real_size = block['size'] * res * 0.9
            marker.scale.x = real_size
            marker.scale.y = real_size
            marker.scale.z = 0.05

            current_score = block.get('dynamic_score', block['base_score'])

            if block['base_score'] < -300:
                marker.color = ColorRGBA(r=0.1, g=0.1, b=0.1, a=0.2)
            else:
                normalized = (current_score - min_score) / score_range
                marker.color = self.get_heatmap_color(normalized)
            
            markers.markers.append(marker)

        self.debug_pub.publish(markers)

    def generate_random_point(self, block):
        gx = block['gx']
        gy = block['gy']
        block_size = block['size']
        
        res = self.map.resolution
        
        cx = block['cx']
        cy = block['cy']
        
        half_size_meters = (block_size * res) / 2.0
        
        rand_x = random.uniform(cx - half_size_meters, cx + half_size_meters)
        rand_y = random.uniform(cy - half_size_meters, cy + half_size_meters)
        rand_theta = random.uniform(0, 360)
        
        return create_nav_goal_at(rand_x, rand_y, rand_theta)

    # =========================================================================
    #                               VICTIMS
    # =========================================================================
    def marker_callback(self, msg):

        # Ignore markers until localization and mission have started
        if self.state == States.STARTUP or not self.amcl_ready or not self.map_ready:
            return
                
        # Ignore if already busy with a victim or returning
        if self.state in [
            States.VICTIM_INTERACTION,
            States.RETURNING,
            States.ALIGNING_FOR_PHOTO,
        ]:
            return

        # Also ignore if we already have a pending ArUco goal
        if self.pending_aruco_id is not None:
            return
        
        self.get_logger().info(f"Found ArUco with state {self.state}. Approaching...")

        for i, pose in enumerate(msg.poses):
            try:
                ps = PoseStamped(header=msg.header, pose=pose)

                if msg.header.frame_id != "map":
                    transformed = self.buffer.transform(ps, "map")
                    marker_pose_in_map = transformed.pose
                else:
                    marker_pose_in_map = ps.pose

                vid = msg.marker_ids[i]

                # Skip if this marker is too close to an existing victim
                if not self.within_two(marker_pose_in_map, threshold=0.2):
                    self.get_logger().info(
                        f"State: {self.state} ArUco {vid} is within 0.2 m of an existing victim, treating as duplicate."
                    )
                    continue

                # We are actually going to handle this victim
                self.get_logger().info(
                    f"Found NEW ArUco {vid} in state {self.state}. Sending photo-alignment goal."
                )

                # Stop whatever navigation was happening
                self.stop_navigation()

                # Record pending victim
                self.pending_aruco_id = vid
                self.pending_aruco_pose = marker_pose_in_map

                # Delegate everything else to the alignment helper
                self.send_photo_alignment_goal()
                return

            except Exception as e:
                self.get_logger().warn(f"Marker callback error: {e}")
                continue


    def send_photo_alignment_goal(self):

        self.victim_capture_in_progress = False

        if self.current_pose is None or self.pending_aruco_pose is None:
            self.get_logger().warn("No pose for alignment, capturing anyway.")
            self.finalize_victim_capture()
            return

        marker = self.pending_aruco_pose

        # Extract positions
        mx = marker.position.x
        my = marker.position.y
        rx = self.current_pose.position.x
        ry = self.current_pose.position.y

        # Face the marker
        heading_to_marker = math.atan2(my - ry, mx - rx)
        desired_yaw = heading_to_marker
        desired_yaw_deg = math.degrees(desired_yaw)

        # Goal 0.6 m in front of marker along that same direction
        dist = 0.6
        goal_x = mx - dist * math.cos(heading_to_marker)
        goal_y = my - dist * math.sin(heading_to_marker)

        self.get_logger().info(
            f"Photo-alignment:\n"
            f"  marker at ({mx:.2f}, {my:.2f})\n"
            f"  goal   at ({goal_x:.2f}, {goal_y:.2f}) yaw={desired_yaw_deg:.1f}°"
        )

        # Build Nav2 goal
        align_goal = create_nav_goal_at(goal_x, goal_y, desired_yaw_deg)

        self.set_state(States.ALIGNING_FOR_PHOTO)

        self.current_goal = align_goal
        self.goal_seq += 1
        goal_seq = self.goal_seq

        fut = self.ac.send_goal_async(align_goal)
        fut.add_done_callback(
            lambda future, seq=goal_seq: self.goal_response_callback(future, seq)
        )



    
    def calculate_approach_pose(self, marker_pose, dist_meters=0.6):
        """
        Compute a goal directly in front of the marker along the line
        from the robot to the marker.

        Steps:
        1) Compute heading from robot -> marker.
        2) Place goal dist_meters in front of the marker along that line.
        3) Orient robot so it looks straight at the marker.
        """
        if self.current_pose is None:
            self.get_logger().warn("Current pose unknown, cannot calc approach vector.")
            # Fallback  stand on top of the marker and face arbitrary direction
            return create_nav_goal_at(marker_pose.position.x, marker_pose.position.y, 0.0)

        # Robot and marker positions
        rx = self.current_pose.position.x
        ry = self.current_pose.position.y
        mx = marker_pose.position.x
        my = marker_pose.position.y

        # Heading from robot to marker (in map frame)
        heading_to_marker = math.atan2(my - ry, mx - rx)

        # Place the goal dist_meters in front of the marker along this line
        goal_x = mx - dist_meters * math.cos(heading_to_marker)
        goal_y = my - dist_meters * math.sin(heading_to_marker)

        # Robot should look toward the marker
        goal_yaw = heading_to_marker  # face directly toward marker
        goal_yaw_deg = math.degrees(goal_yaw)

        self.get_logger().info(
            f"Approach pose for ArUco: marker=({mx:.2f}, {my:.2f}), "
            f"goal=({goal_x:.2f}, {goal_y:.2f}), yaw={goal_yaw_deg:.1f} deg"
        )

        return create_nav_goal_at(goal_x, goal_y, goal_yaw_deg)

    def finalize_victim_capture(self):
        """Publish Victim message and store victim once we are aligned and ready."""
        if self.pending_aruco_id is None or self.pending_aruco_pose is None:
            self.get_logger().warn("finalize_victim_capture called with no pending ArUco.")
            return

        self.get_logger().info(f"Capturing victim {self.pending_aruco_id}...")

        v_msg = Victim()
        v_msg.id = self.pending_aruco_id
        v_msg.point = PointStamped()
        v_msg.point.header.frame_id = "map"
        v_msg.point.point = self.pending_aruco_pose.position

        if self.latest_image is not None:
            v_msg.image = self.latest_image

        self.victim_pub.publish(v_msg)

        if self.latest_image is not None:
            self.victim_image_pub.publish(self.latest_image)
            try:
                cv_img = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"/tmp/victim_{self.pending_aruco_id}_{ts}.png"
                cv2.imwrite(filename, cv_img)
                self.get_logger().info(f"Saved victim image to {filename}")
            except CvBridgeError as e:
                self.get_logger().warn(f"Failed to convert victim image: {e}")
        else:
            self.get_logger().warn("No camera frame available for victim photo.")

        # Deduplicate based on location
        if not self.within_two(self.pending_aruco_pose, threshold=0.2):
            self.get_logger().info(
                f"Victim {self.pending_aruco_id} is very close to an existing victim. Skipping new record."
            )
        else:
            key = (
                self.pending_aruco_pose.position.x,
                self.pending_aruco_pose.position.y,
                self.pending_aruco_pose.position.z,
                self.pending_aruco_id,
            )
            self.victims[key] = {
                "visited": True,
                "pose": self.pending_aruco_pose,
                "image": self.latest_image,
            }

        # Clear pending victim and resume mission
        self.pending_aruco_id = None
        self.pending_aruco_pose = None
        self.set_state(States.DECIDING)
        self.send_goal()

    def start_photo_alignment(self):
        """Begin fine alignment rotation so the robot faces the marker exactly."""
        if self.current_pose is None or self.pending_aruco_pose is None:
            self.get_logger().warn("Cannot start alignment: missing pose. Capturing anyway.")
            self.finalize_victim_capture()
            return

        if self.photo_align_timer is not None:
            # Already aligning
            return

        self.get_logger().info("Starting fine alignment before photo capture.")
        self.photo_align_started = True
        # Run at 20 Hz
        self.photo_align_timer = self.create_timer(0.05, self.photo_align_step)

    def photo_align_step(self):
        """Timer callback that rotates the robot to face the pending ArUco."""
        if self.current_pose is None or self.pending_aruco_pose is None:
            self.get_logger().warn("Alignment step without valid pose, capturing anyway.")
            self.stop_rotation_and_capture()
            return

        # Robot pose
        rx = self.current_pose.position.x
        ry = self.current_pose.position.y
        q = self.current_pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

        # Marker pose
        mx = self.pending_aruco_pose.position.x
        my = self.pending_aruco_pose.position.y

        # Desired yaw: face marker
        desired_yaw = math.atan2(my - ry, mx - rx)

        # Smallest angle difference
        yaw_err = math.atan2(math.sin(desired_yaw - yaw), math.cos(desired_yaw - yaw))
        yaw_err_deg = math.degrees(yaw_err)

        # Threshold for "good enough" alignment
        align_thresh_rad = math.radians(5.0)

        if abs(yaw_err) > align_thresh_rad:
            # Rotate toward the marker
            twist = TwistStamped()
            twist.header.stamp = self.get_clock().now().to_msg()
            twist.twist.angular.z = 0.4 if yaw_err > 0 else -0.4
            self.cmd_pub.publish(twist)
            self.get_logger().info(
                f"Aligning to victim: yaw_err={yaw_err_deg:.1f} deg, rotating..."
            )
        else:
            self.get_logger().info(
                f"Alignment complete: yaw_err={yaw_err_deg:.1f} deg. Capturing photo."
            )
            self.stop_rotation_and_capture()

    def stop_rotation_and_capture(self):
        """Stop turning and perform the actual victim capture."""
        # Stop rotation
        stop_msg = TwistStamped()
        stop_msg.header.stamp = self.get_clock().now().to_msg()
        self.cmd_pub.publish(stop_msg)

        # Stop the timer
        if self.photo_align_timer is not None:
            self.photo_align_timer.cancel()
            self.photo_align_timer = None

        self.photo_align_started = False

        # Now actually capture and publish victim
        self.finalize_victim_capture()


    def within_two(self, victim, threshold=0.2):
        for unique in self.victims.keys():
            if (abs(unique[0] - victim.position.x) <= threshold) and \
               (abs(unique[1] - victim.position.y) <= threshold) and \
               (abs(unique[2] - victim.position.z) <= threshold):
                return False 
        return True

    def get_unique_victims(self):
        return [k for k, v in self.victims.items() if v is False]

############# Helpers ##############

def create_nav_goal_at(x, y, theta_deg):
    theta = math.radians(theta_deg)
    goal = NavigateToPose.Goal()
    goal.pose.header.frame_id = 'map'
    goal.pose.pose.position.x = x
    goal.pose.pose.position.y = y
    q = tf_transformations.quaternion_from_euler(0, 0, theta)
    goal.pose.pose.orientation.x, goal.pose.pose.orientation.y, goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = q
    return goal

def compute_distance_to_start(start_pose, current_pose):
    if start_pose is None or current_pose is None:
        print("No pose yet, cannot compute distance.")
        return None
    x1, y1 = start_pose.position.x, start_pose.position.y
    x2, y2 = current_pose.position.x, current_pose.position.y

    return abs(x2 - x1) + abs(y2 - y1)

def estimate_return_time(start_pose, current_pose, speed_mps=0.125):
    distance = compute_distance_to_start(start_pose, current_pose)
    if distance is None:
        return None

    return distance / speed_mps

def create_random_free_goal(map):
    if map is None:
        print("create_random_free_goal() called before map loaded!")
        return create_nav_goal_at(0.0, 0.0, 0.0)

    x_min = map.origin_x
    x_max = map.origin_x + map.width * map.resolution
    y_min = map.origin_y
    y_max = map.origin_y + map.height * map.resolution

    while True:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        mx = int((x - map.origin_x) / map.resolution)
        my = int((y - map.origin_y) / map.resolution)

        if mx < 0 or mx >= map.width or my < 0 or my >= map.height:
            continue

        occ = map.grid[my, mx]

        if occ != 0:
            continue

        theta_deg = random.uniform(0, 360)
        print(f"Generated FREE goal at ({x:.2f}, {y:.2f}), occ={occ}")
        return create_nav_goal_at(x, y, theta_deg)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Navigate to random goals."
    )

    rclpy.init()
    node = RandomNavNode()

    future = node.future_event

    rclpy.spin_until_future_complete(node, future)

    node.destroy_node()
    rclpy.shutdown()
