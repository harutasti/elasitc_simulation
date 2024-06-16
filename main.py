import pybullet as p
import pybullet_data
import time
import numpy as np

# シミュレーションの初期化
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 重力の設定
p.setGravity(0, 0, -9.8)

# 地面の追加
plane_id = p.loadURDF("plane.urdf")

# 弾性体を構成する粒子のパラメータ
particle_radius = 0.05
particle_mass = 1.0
num_particles_x = 5
num_particles_y = 5

# 粒子の初期配置
particle_ids = []
positions = []
for i in range(num_particles_x):
    for j in range(num_particles_y):
        pos = [i * 2 * particle_radius, j * 2 * particle_radius, 1]
        particle_id = p.createCollisionShape(p.GEOM_SPHERE, radius=particle_radius)
        particle_ids.append(p.createMultiBody(particle_mass, particle_id, -1, pos))
        positions.append(pos)

# バネとダンパーの定数
spring_constant = 500
damping_coefficient = 10

# 初期距離行列の計算
initial_distances = np.zeros((num_particles_x * num_particles_y, num_particles_x * num_particles_y))
for i in range(num_particles_x * num_particles_y):
    for j in range(num_particles_x * num_particles_y):
        if i != j:
            dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
            initial_distances[i][j] = dist


def apply_spring_damper_forces():
    for i in range(num_particles_x * num_particles_y):
        for j in range(i + 1, num_particles_x * num_particles_y):
            pos_i = np.array(p.getBasePositionAndOrientation(particle_ids[i])[0])
            pos_j = np.array(p.getBasePositionAndOrientation(particle_ids[j])[0])
            vel_i = np.array(p.getBaseVelocity(particle_ids[i])[0])
            vel_j = np.array(p.getBaseVelocity(particle_ids[j])[0])

            distance = np.linalg.norm(pos_i - pos_j)
            if distance < 2 * particle_radius:
                initial_distance = initial_distances[i][j]
                spring_force_magnitude = spring_constant * (distance - initial_distance)
                damping_force_magnitude = damping_coefficient * np.dot(vel_j - vel_i, (pos_j - pos_i) / distance)

                total_force = (spring_force_magnitude + damping_force_magnitude) * (pos_j - pos_i) / distance

                p.applyExternalForce(particle_ids[i], -1, total_force, pos_i.tolist(), p.WORLD_FRAME)
                p.applyExternalForce(particle_ids[j], -1, -total_force, pos_j.tolist(), p.WORLD_FRAME)


# シミュレーションの実行
while p.isConnected():
    apply_spring_damper_forces()
    p.stepSimulation()
    time.sleep(1. / 240.)
