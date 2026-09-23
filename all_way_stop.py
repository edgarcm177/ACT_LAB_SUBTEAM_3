"""
Simple MPC + FIFO Controller for an All-Way-Stop Intersection
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches 
from scipy.optimize import minimize


# =============================================================================
# 1. PARAMETERS
# =============================================================================

DT = 0.20                 # sampling time [s]
SIM_TIME = 35.0           # maximum simulation time [s]
N = 20                    # prediction horizon
V_DES = 8.0               # desired speed in GO [m/s]

STOP_LINE = 0.0           # stop line position [m]
INTERSECTION_EXIT = 12.0  # intersection is clear after this point [m]
DELETE_POINT = 25.0       # remove vehicle after this point [m]

STOP_POS_TOL = 0.50       # stopping position tolerance [m]
STOP_SPEED_TOL = 0.20     # stopping speed tolerance [m/s]

# Objective-function weights from the report
W_POSITION = 100.0
W_SPEED = 150.0
W_ACCEL = 0.20
W_CROSS = 2000.0

MAXITER = 60


# =============================================================================
# 2. CAR
# =============================================================================

class Car:
    def __init__(self, car_id, approach, position, speed):
        self.id = car_id
        self.approach = approach

        self.p = position
        self.v = speed

        self.state = "APPROACH"

        self.stop_time = None
        self.priority = None

        # Initial guesses for the two optimization problems.
        # Each guess contains N acceleration values.
        self.stop_guess = np.zeros(N)
        self.go_guess = np.zeros(N)

        # Data saved for plots.
        self.time_history = []
        self.position_history = []
        self.speed_history = []
        self.acceleration_history = []
        self.state_history = []


# =============================================================================
# 3. CONTROLLER
# =============================================================================

class Controller:
    def __init__(self):
        self.queue = []
        self.active_cars = []
        self.next_priority = 1

    # -------------------------------------------------------------------------
    # Vehicle prediction model
    # -------------------------------------------------------------------------
    def predict(self, p0, v0, acceleration_sequence):

        p = np.zeros(N + 1)
        v = np.zeros(N + 1)

        p[0] = p0
        v[0] = v0

        for k in range(N):
            a = acceleration_sequence[k]

            p[k + 1] = p[k] + v[k] * DT + 0.5 * a * DT**2
            v[k + 1] = v[k] + a * DT

        return p, v

    # -------------------------------------------------------------------------
    # Problem 1: STOP
    # -------------------------------------------------------------------------
    def stop_cost(self, acceleration_sequence, car):
        """
        J_stop = W_POSITION * p(N)^2
               + W_SPEED    * v(N)^2
               + sum[ W_ACCEL * a(k)^2
                    + W_CROSS * max(p(k+1), 0)^2 ]
        """

        p, v = self.predict(car.p, car.v, acceleration_sequence)

        cost = W_POSITION * p[N]**2
        cost += W_SPEED * v[N]**2

        for k in range(N):
            cost += W_ACCEL * acceleration_sequence[k]**2
            cost += W_CROSS * max(p[k + 1], 0.0)**2

        return cost

    def solve_stop_problem(self, car):
        """Solve the STOP problem and return the first optimal acceleration."""

        result = minimize(
            self.stop_cost,
            car.stop_guess,
            args=(car,),
            method="BFGS",
            options={"maxiter": MAXITER, "disp": False},
        )

        optimal_sequence = result.x
        car.stop_guess = optimal_sequence

        # MPC: apply only the first input.
        return optimal_sequence[0]

    # -------------------------------------------------------------------------
    # Problem 2: GO
    # -------------------------------------------------------------------------
    def go_cost(self, acceleration_sequence, car):
        """
        J_go = sum[ W_SPEED * (v(k+1) - V_DES)^2
                  + W_ACCEL * a(k)^2 ]
        """

        _, v = self.predict(car.p, car.v, acceleration_sequence)

        cost = 0.0

        for k in range(N):
            cost += W_SPEED * (v[k + 1] - V_DES)**2
            cost += W_ACCEL * acceleration_sequence[k]**2

        return cost

    def solve_go_problem(self, car):
        """Solve the GO problem and return the first optimal acceleration."""

        result = minimize(
            self.go_cost,
            car.go_guess,
            args=(car,),
            method="BFGS",
            options={"maxiter": MAXITER, "disp": False},
        )

        optimal_sequence = result.x
        car.go_guess = optimal_sequence

        # MPC: apply only the first input.
        return optimal_sequence[0]

    # -------------------------------------------------------------------------
    # FIFO logic
    # -------------------------------------------------------------------------
    def register_stopped_car(self, car, time):
        """Add a newly stopped car to the FIFO queue."""

        if car.priority is not None:
            return

        car.stop_time = time
        car.priority = self.next_priority
        self.next_priority += 1

        self.queue.append(car.id)
        car.state = "WAIT"
    ##Helper to determine if two approaches do not conflict
    def can_go_together(self, dir1, dir2):
        return (dir1, dir2) in [("N", "S"), ("S", "N"), ("E", "W"), ("W", "E")]

    #Added logic for checking if second car in queue can also go
    def release_next_car(self, cars):
        """If the intersection is free, allow the first car in the queue to go."""

        if len(self.active_cars) > 0:
            return

        if len(self.queue) == 0:
            return

        first_id = self.queue.pop(0)
        first_approach = ""

        for car in cars:
            if car.id == first_id:
                car.state = "GO"
                self.active_cars.append(car.id)
                first_approach = car.approach
                break

        if len(self.queue) > 0:
            second_id = self.queue[0]
            second_approach = ""

            for car in cars:
                if car.id == second_id:
                    second_approach = car.approach
                    break

            if self.can_go_together(first_approach, second_approach):
                self.queue.pop(0)  # Remove the second car from the queue
                for car in cars:
                    if car.id == second_id:
                        car.state = "GO"
                        self.active_cars.append(car.id)
                        break

    def check_intersection(self, cars):
        """Free the intersection after the active car reaches the exit point."""

        if len(self.active_cars) == 0:
            return

        # Iterate backwards to safely remove items from the list during the loop
        for i in range(len(self.active_cars) - 1, -1, -1):
            current_id = self.active_cars[i]
            
            for car in cars:
                if car.id == current_id:
                    if car.p >= INTERSECTION_EXIT:
                        self.active_cars.pop(i)
                    break


# =============================================================================
# 4. DEMONSTRATION VEHICLES
# =============================================================================

def create_cars():
    return [
        Car(1, "W", -24.0, 7.0),
        Car(2, "E", -30.0, 7.5),
        Car(3, "S", -36.0, 8.0),
        Car(4, "N", -42.0, 8.0),
    ]


# =============================================================================
# 5. SIMULATION
# =============================================================================

def run_simulation():
    cars = create_cars()
    controller = Controller()

    times = np.arange(0.0, SIM_TIME + DT, DT)

    for time in times:

        # ---------------------------------------------------------------------
        # A. Check whether APPROACH cars have stopped.
        # ---------------------------------------------------------------------
        # Cars are checked in ID order. If two stop at exactly the same time,
        # the smaller ID receives priority first.
        for car in cars:
            if car.state != "APPROACH":
                continue

            close_to_stop_line = abs(car.p - STOP_LINE) <= STOP_POS_TOL
            almost_stopped = car.v <= STOP_SPEED_TOL

            if close_to_stop_line and almost_stopped:
                car.p = STOP_LINE
                car.v = 0.0
                controller.register_stopped_car(car, time)

        # ---------------------------------------------------------------------
        # B. If possible, release the first car in the FIFO queue.
        # ---------------------------------------------------------------------
        controller.release_next_car(cars)

        # ---------------------------------------------------------------------
        # C. Compute one acceleration to apply to each car.
        # ---------------------------------------------------------------------
        accelerations = {}

        for car in cars:
            if car.state == "APPROACH":
                accelerations[car.id] = controller.solve_stop_problem(car)

            elif car.state == "WAIT":
                accelerations[car.id] = 0.0

            elif car.state == "GO":
                accelerations[car.id] = controller.solve_go_problem(car)

            else:  # DONE
                accelerations[car.id] = 0.0

        # ---------------------------------------------------------------------
        # D. Save the current state and move each car one simulation step.
        # ---------------------------------------------------------------------
        for car in cars:
            if car.state == "DONE":
                continue

            a = accelerations[car.id]

            car.time_history.append(time)
            car.position_history.append(car.p)
            car.speed_history.append(car.v)
            car.acceleration_history.append(a)
            car.state_history.append(car.state)

            if car.state == "WAIT":
                car.p = STOP_LINE
                car.v = 0.0

            else:
                car.p = car.p + car.v * DT + 0.5 * a * DT**2
                car.v = car.v + a * DT

        # ---------------------------------------------------------------------
        # E. Check whether the active car has cleared the intersection.
        # ---------------------------------------------------------------------
        controller.check_intersection(cars)

        # ---------------------------------------------------------------------
        # F. Remove cars after the downstream deletion point.
        # ---------------------------------------------------------------------
        for car in cars:
            if car.state == "GO" and car.p >= DELETE_POINT:
                car.state = "DONE"

        if all(car.state == "DONE" for car in cars):
            break

    return cars


# =============================================================================
# 6. RESULTS
# =============================================================================

def print_summary(cars):
    print("\nFIFO crossing order")
    print("-------------------")

    ordered_cars = sorted(cars, key=lambda car: car.priority)

    for car in ordered_cars:
        print(
            f"Priority {car.priority}: "
            f"Vehicle {car.id} ({car.approach}), "
            f"stop time = {car.stop_time:.1f} s"
        )


def plot_results(cars):
    # Position plot
    plt.figure(figsize=(9, 5))

    for car in cars:
        plt.plot(
            car.time_history,
            car.position_history,
            label=f"Vehicle {car.id} ({car.approach})",
        )

    plt.axhline(STOP_LINE, linestyle="--", linewidth=1, label="Stop line")
    plt.axhline(
        INTERSECTION_EXIT,
        linestyle=":",
        linewidth=1,
        label="Intersection exit",
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("Vehicle Position")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("position_vs_time.png", dpi=200)
    plt.close()

    # Speed plot
    plt.figure(figsize=(9, 5))

    for car in cars:
        plt.plot(
            car.time_history,
            car.speed_history,
            label=f"Vehicle {car.id} ({car.approach})",
        )

    plt.axhline(V_DES, linestyle="--", linewidth=1, label="Desired speed")

    plt.xlabel("Time [s]")
    plt.ylabel("Speed [m/s]")
    plt.title("Vehicle Speed")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("speed_vs_time.png", dpi=200)
    plt.close()



#Animation Function
def animate_results(cars):
    max_frames = max(len(car.time_history) for car in cars)
    
    fig, ax = plt.subplots(figsize=(12, 12), dpi=120)
    ax.set_xlim(-45, 45)
    ax.set_ylim(-45, 45)
    ax.set_aspect('equal')
    ax.set_facecolor('#2e2e2e')
    ax.set_title("All-Way Stop MPC Controller", color='white', pad=20, fontsize=16)

    road_width = 12
    intersection = patches.Rectangle((0, 0), road_width, road_width, color='#555555', zorder=1)
    ax.add_patch(intersection)

    ax.add_patch(patches.Rectangle((-50, 0), 50, road_width, color='#444444', zorder=0))  
    ax.add_patch(patches.Rectangle((12, 0), 50, road_width, color='#444444', zorder=0))   
    ax.add_patch(patches.Rectangle((0, -50), road_width, 50, color='#444444', zorder=0))  
    ax.add_patch(patches.Rectangle((0, 12), road_width, 50, color='#444444', zorder=0))   

    ax.plot([-50, 0], [6, 6], color='#e6c800', linestyle='--', zorder=2)
    ax.plot([12, 50], [6, 6], color='#e6c800', linestyle='--', zorder=2)
    ax.plot([6, 6], [-50, 0], color='#e6c800', linestyle='--', zorder=2)
    ax.plot([6, 6], [12, 50], color='#e6c800', linestyle='--', zorder=2)

    ax.plot([0, 0], [0, 6], color='white', linewidth=4, zorder=2)       
    ax.plot([12, 12], [6, 12], color='white', linewidth=4, zorder=2)    
    ax.plot([6, 12], [0, 0], color='white', linewidth=4, zorder=2)      
    ax.plot([0, 6], [12, 12], color='white', linewidth=4, zorder=2)     

    STATE_COLORS = {
        "APPROACH": "#f39c12",  
        "WAIT": "#e74c3c",      
        "GO": "#2ecc71"         
    }

    def get_xy(approach, p):
        if approach == "W": return (p, 3)         
        if approach == "E": return (12 - p, 9)    
        if approach == "S": return (9, p)         
        if approach == "N": return (3, 12 - p)    
        return (0, 0)

    car_length = 4.5
    car_width = 2.0
    car_patches = {}
    car_texts = {}

    for car in cars:
        rect = patches.Rectangle((0, 0), car_length, car_width, color='white', zorder=3)
        ax.add_patch(rect)
        car_patches[car.id] = rect
        
        text = ax.text(0, 0, f"V{car.id}", color='black', weight='bold', fontsize=8,
                       ha='center', va='center', zorder=4)
        car_texts[car.id] = text

    # Telemetry coordinates mapped exactly to top-left quadrant
    time_text = ax.text(-40, 40, '', color='white', fontsize=14, fontfamily='monospace')
    telem_text = ax.text(-40, 32, '', color='white', fontsize=10, fontfamily='monospace', va='top')

    # Fetch DT from the global scope of the file
    global DT

    def update(frame_idx):
        current_time = frame_idx * DT
        time_text.set_text(f"Simulation Time: {current_time:05.2f} s")
        telem_lines = ["TELEMETRY:"]
        
        for car in sorted(cars, key=lambda c: c.id):
            vid = car.id
            approach = car.approach
            
            if frame_idx < len(car.position_history):
                p = car.position_history[frame_idx]
                v = car.speed_history[frame_idx]
                state = car.state_history[frame_idx]
            else:
                state = "DONE"
                p = car.position_history[-1] if car.position_history else 0
                v = 0.0
                
            cx, cy = get_xy(approach, p)
            rect = car_patches[vid]
            
            if state != "DONE":
                telem_lines.append(f"V{vid}({approach}): {state:<8} | p={p:>5.1f} | v={v:>4.1f}")
            
            if state == "DONE" or cx < -45 or cx > 45 or cy < -45 or cy > 45:
                rect.set_alpha(0.0)
                car_texts[vid].set_alpha(0.0)
            else:
                rect.set_alpha(1.0)
                car_texts[vid].set_alpha(1.0)
                rect.set_facecolor(STATE_COLORS.get(state, "white"))
                
                if approach == "W":
                    rect.set_width(car_length)
                    rect.set_height(car_width)
                    rect.set_xy((cx - car_length, cy - car_width/2))
                    car_texts[vid].set_position((cx - car_length/2, cy))
                elif approach == "E":
                    rect.set_width(car_length)
                    rect.set_height(car_width)
                    rect.set_xy((cx, cy - car_width/2))
                    car_texts[vid].set_position((cx + car_length/2, cy))
                elif approach == "S":
                    rect.set_width(car_width)
                    rect.set_height(car_length)
                    rect.set_xy((cx - car_width/2, cy - car_length))
                    car_texts[vid].set_position((cx, cy - car_length/2))
                elif approach == "N":
                    rect.set_width(car_width)
                    rect.set_height(car_length)
                    rect.set_xy((cx - car_width/2, cy))
                    car_texts[vid].set_position((cx, cy + car_length/2))

        telem_text.set_text("\n".join(telem_lines))
        return list(car_patches.values()) + list(car_texts.values()) + [time_text, telem_text]

    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=200, blit=True)
    plt.tight_layout()
    plt.show()

# =============================================================================
# 7. MAIN
# =============================================================================

if __name__ == "__main__":
    cars = run_simulation()
    print_summary(cars)
    plot_results(cars)
    animate_results(cars)
