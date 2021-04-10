import pybullet as p
#import pybullet_data as pd
import time

p.connect(p.GUI)
dt = p.getPhysicsEngineParameters()['fixedTimeStep']


col_shape_id = p.createCollisionShape(
    shapeType=p.GEOM_MESH,
    fileName="tablebody.stl",
)

viz_shape_id = p.createVisualShape(
    shapeType=p.GEOM_MESH,
    fileName="tablebody.stl",
)

body_id = p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=col_shape_id,
    baseVisualShapeIndex=viz_shape_id,
    basePosition=(0, 0, 0),
    baseOrientation=(0, 0, 0, 1),
)


while p.isConnected():
  p.stepSimulation()
  time.sleep(dt)

