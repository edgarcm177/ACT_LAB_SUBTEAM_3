"""Experimental 2D MPC comparing ellipse and superellipse collision potentials.
Run python all_way_stop.py [--no-animation]. See EXPERIMENT.md.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, patches
from scipy.optimize import minimize

DT, SIM_TIME, N, V_DES = 0.2, 35.0, 20, 8.0
W_SPEED, W_ACCEL = 8.0, 0.2
W_LANE, W_LATERAL_SPEED, W_ROAD, W_COLLISION = 12.0, 10.0, 5000.0, 2000.0
SAFETY_LONG, SAFETY_LAT, SHARPNESS = 7.0, 3.5, 5.0
SHAPE_POWER = 4  # 2: ellipse baseline; 4: rounded rectangle experiment
MAX_ACCEL, MAX_BRAKE, MAX_LATERAL_ACCEL = 2.5, 4.0, 1.5
MAX_SPEED, MAXITER, LANE_DEVIATION = 10.0, 60, 1.8


class Car:
    def __init__(self, car_id, approach, position, speed):
        self.id, self.approach = car_id, approach
        # Centers when the front bumper reaches the stop line.
        origin, forward = {
            'W': ((-2.25, 3.), (1., 0.)), 'E': ((14.25, 9.), (-1., 0.)),
            'S': ((9., -2.25), (0., 1.)), 'N': ((3., 14.25), (0., -1.))
        }[approach]
        self.origin, self.forward = np.array(origin), np.array(forward)
        self.left = np.array([-self.forward[1], self.forward[0]])
        self.xy = self.origin + position * self.forward
        self.velocity = speed * self.forward
        self.guess = np.zeros((N, 2))
        self.state, self.stop_time = 'APPROACH', None
        self.time_history, self.xy_history = [], []
        self.velocity_history, self.acceleration_history = [], []

    @property
    def p(self):
        return float((self.xy - self.origin) @ self.forward)


class Controller:
    def __init__(self):
        self.failed_solves = 0

    def predict(self, x0, y0, vx0, vy0, acceleration_sequence):
        """World-coordinate dynamics, returning x, y, vx, vy trajectories."""
        a = np.asarray(acceleration_sequence).reshape(N, 2)
        v = np.vstack(([vx0, vy0], [vx0, vy0] + DT * np.cumsum(a, axis=0)))
        xy = np.vstack(([x0, y0], [x0, y0] + np.cumsum(
            v[:-1] * DT + 0.5 * a * DT**2, axis=0)))
        return xy[:, 0], xy[:, 1], v[:, 0], v[:, 1]

    def trajectory(self, car, local_accel):
        world = local_accel[:, :1] * car.forward + local_accel[:, 1:] * car.left
        x, y, vx, vy = self.predict(*car.xy, *car.velocity, world)
        return np.column_stack((x, y)), np.column_stack((vx, vy))

    @staticmethod
    def collision_potential(relative_xy, obstacle):
        """q=1 is a superellipse; exponent 2 recovers the ellipse baseline."""
        q = np.abs((relative_xy @ obstacle.forward) / SAFETY_LONG)**SHAPE_POWER
        q += np.abs((relative_xy @ obstacle.left) / SAFETY_LAT)**SHAPE_POWER
        return (np.logaddexp(0., SHARPNESS * (1. - q)) / SHARPNESS)**2

    def objective(self, flat, car, obstacles):
        a = flat.reshape(N, 2)
        xy, velocity = self.trajectory(car, a)
        p = (xy[1:] - car.origin) @ car.forward
        lateral = (xy[1:] - car.origin) @ car.left
        vf, vl = velocity[1:] @ car.forward, velocity[1:] @ car.left
        if car.state == 'APPROACH':
            cost = 100. * p[-1]**2 + 150. * vf[-1]**2
            cost += 2000. * np.sum(np.maximum(p, 0.)**2)
        else:
            cost = W_SPEED * np.sum((vf - V_DES)**2)
        cost += W_ACCEL * np.sum(a**2)
        cost += W_LATERAL_SPEED * np.sum(vl**2) + W_LANE * np.sum(lateral**2)
        cost += W_ROAD * np.sum(np.maximum(np.abs(lateral) - LANE_DEVIATION, 0.)**2)
        for other, prediction in obstacles:
            cost += W_COLLISION * np.sum(self.collision_potential(xy[1:] - prediction[1:], other))
        return float(cost)

    def solve(self, car, obstacles):
        initial_speed = car.velocity @ car.forward

        def constraints(flat):
            vf = initial_speed + DT * np.cumsum(flat.reshape(N, 2)[:, 0])
            return np.concatenate((vf, MAX_SPEED - vf))

        result = minimize(self.objective, car.guess.ravel(), args=(car, obstacles),
                          method='SLSQP', bounds=[(-MAX_BRAKE, MAX_ACCEL),
                          (-MAX_LATERAL_ACCEL, MAX_LATERAL_ACCEL)] * N,
                          constraints={'type': 'ineq', 'fun': constraints},
                          options={'maxiter': MAXITER, 'ftol': 1e-4})
        if (not result.success or not np.all(np.isfinite(result.x))
                or np.min(constraints(result.x)) < -1e-5):
            self.failed_solves += 1
            plan, speed = np.zeros((N, 2)), initial_speed
            for k in range(N):
                plan[k, 0] = -min(MAX_BRAKE, max(speed, 0.) / DT)
                speed += DT * plan[k, 0]
        else:
            plan = result.x.reshape(N, 2)
        car.guess = np.vstack((plan[1:], plan[-1:]))
        return plan[0].copy()


def create_cars():
    return [Car(1, 'W', -24., 7.), Car(2, 'E', -30., 7.5),
            Car(3, 'S', -36., 8.), Car(4, 'N', -42., 8.)]


def run_simulation():
    cars, controller = create_cars(), Controller()
    minimum_gap = float('inf')
    for time in np.arange(0., SIM_TIME + DT, DT):
        active = [c for c in cars if c.state != 'DONE']
        for car in active:
            if (car.state == 'APPROACH' and abs(car.p) <= 0.5
                    and np.linalg.norm(car.velocity) <= 0.2):
                car.state, car.stop_time = 'GO', time
        # Freeze the previous shifted plans before solving any new controls.
        predictions = {c.id: controller.trajectory(c, c.guess)[0] for c in active}
        actions = {c.id: controller.solve(c, [(o, predictions[o.id]) for o in active
                                             if o.id != c.id]) for c in active}
        for car in active:
            a = actions[car.id]
            car.time_history.append(time)
            car.xy_history.append(car.xy.copy())
            car.velocity_history.append(car.velocity.copy())
            car.acceleration_history.append(a.copy())
            world_a = a[0] * car.forward + a[1] * car.left
            car.xy += car.velocity * DT + 0.5 * world_a * DT**2
            car.velocity += world_a * DT
        for i, car in enumerate(active):
            size = np.abs(car.forward) * 4.5 + np.abs(car.left) * 2.
            for other in active[i + 1:]:
                other_size = np.abs(other.forward) * 4.5 + np.abs(other.left) * 2.
                gap = np.abs(car.xy - other.xy) - (size + other_size) / 2
                minimum_gap = min(minimum_gap, float(np.max(gap)))
        for car in active:
            if car.state == 'GO' and car.p >= 25.:
                car.state = 'DONE'
        if all(c.state == 'DONE' for c in cars):
            break
    print(f'Solver fallbacks: {controller.failed_solves}')
    print(f'Minimum signed rectangle separation at sampled steps: {minimum_gap:.2f} m')
    if minimum_gap < 0:
        print('WARNING: vehicle bodies overlapped at a sampled step.')
    for car in cars:
        print(f'Vehicle {car.id} ({car.approach}): stop={car.stop_time}, final={car.state}')
    return cars


def plot_results(cars):
    for filename, ylabel, values in [
        ('position_vs_time.png', 'Longitudinal position [m]',
         lambda c: (np.array(c.xy_history) - c.origin) @ c.forward),
        ('speed_vs_time.png', 'Speed [m/s]', lambda c: np.linalg.norm(c.velocity_history, axis=1)),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for car in cars:
            ax.plot(car.time_history, values(car), label=f'V{car.id} ({car.approach})')
        ax.set(xlabel='Time [s]', ylabel=ylabel)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(filename, dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 8))
    for car in cars:
        xy = np.array(car.xy_history)
        ax.plot(xy[:, 0], xy[:, 1], label=f'V{car.id} ({car.approach})')
    ax.set(xlabel='x [m]', ylabel='y [m]', title='Actual 2D trajectories')
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig('trajectories_2d.png', dpi=160)
    plt.close(fig)


def animate_results(cars):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set(xlim=(-45, 45), ylim=(-45, 45), xlabel='x [m]', ylabel='y [m]')
    ax.set_aspect('equal')
    ax.set_facecolor('#303030')
    ax.add_patch(patches.Rectangle((-50, 0), 100, 12, color='#666666'))
    ax.add_patch(patches.Rectangle((0, -50), 12, 100, color='#666666'))
    ax.plot([-50, 50], [6, 6], 'y--', alpha=0.4)
    ax.plot([6, 6], [-50, 50], 'y--', alpha=0.4)
    bodies, labels, contours = [], [], []
    theta = np.linspace(0., 2 * np.pi, 161)
    longitudinal = SAFETY_LONG * np.sign(np.cos(theta)) * np.abs(np.cos(theta))**(2 / SHAPE_POWER)
    lateral = SAFETY_LAT * np.sign(np.sin(theta)) * np.abs(np.sin(theta))**(2 / SHAPE_POWER)
    for car in cars:
        size = np.abs(car.forward) * 4.5 + np.abs(car.left) * 2.
        body = patches.Rectangle((0, 0), *size, color=f'C{car.id - 1}', zorder=3)
        contour, = ax.plot([], [], color=f'C{car.id - 1}', alpha=0.5)
        ax.add_patch(body)
        bodies.append(body)
        contours.append(contour)
        labels.append(ax.text(0, 0, f'V{car.id}', ha='center', va='center', zorder=4))

    def update(frame):
        ax.set_title(f'Superellipse MPC (power={SHAPE_POWER}) | t={frame * DT:.1f} s')
        for car, body, label, contour in zip(cars, bodies, labels, contours):
            visible = frame < len(car.xy_history)
            for artist in (body, label, contour):
                artist.set_visible(visible)
            if visible:
                xy = car.xy_history[frame]
                body.set_xy(xy - np.array([body.get_width(), body.get_height()]) / 2)
                label.set_position(xy)
                outline = xy + longitudinal[:, None] * car.forward + lateral[:, None] * car.left
                contour.set_data(outline[:, 0], outline[:, 1])
        return bodies + labels + contours

    ani = animation.FuncAnimation(fig, update, frames=max(len(c.time_history) for c in cars),
                                  interval=DT * 1000, repeat=False)
    plt.show()
    return ani


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-animation', action='store_true')
    parser.add_argument('--shape-power', type=int, choices=(2, 4), default=SHAPE_POWER,
                        help='2: ellipse baseline; 4: superellipse experiment (default)')
    args = parser.parse_args()
    SHAPE_POWER = args.shape_power
    cars = run_simulation()
    plot_results(cars)
    if not args.no_animation:
        animate_results(cars)
