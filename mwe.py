import drjit as dr
import mitsuba as mi
#mi.set_variant('llvm_ad_rgb')
mi.set_variant('cuda_ad_rgb')
mi.Thread.thread().logger().set_log_level(mi.LogLevel.Debug)
import numpy as np
from utils import *


def render(spp, ltype="spotlight"):

    # construct some poses for the board
    ref_poses = []
    ref_poses = []
    for z in [0.5, 1, 2]:
        for log_rot in [[0.1, 0., 0.],[0., 0.1, 0.],[0., 0., 0.1]]:
            ref_poses.append(scalar_exp_map_SO3XR3(mi.ScalarVector3f([0,0,z]), mi.ScalarVector3f(log_rot)))
    
    ref_imgs = []

    # add a light left of the camera and rotate inwards
    T_light2world = mi.ScalarTransform4f().translate([0.015,0,0]).rotate(axis=[0, 1, 0], angle=-5)

    for i, T_board2world in enumerate(ref_poses): 
        scene = get_scene(spp, ltype, T_light2world, T_board2world)
        params = mi.traverse(scene)
        
        # render
        ref_imgs.append(mi.render(scene, params, sensor=0, spp=spp, seed=42))


        # save image
        mi.util.write_bitmap(f"./img_ref_view_{i}_{ltype}.png", ref_imgs[i])


    return ref_imgs, ref_poses


def optimize(spp, ref_imgs, ref_poses, ltype="spotlight"):

    vertex_key = "board.vertex_positions"
    
    if ltype == "spotlight":
        light_pose_key = "spot.to_world"
    if ltype == "tablight":
        light_pose_key = "tab_spot.to_world"


    scene = get_scene(spp, ltype, mi.ScalarTransform4f(), ref_poses[0])
    params = mi.traverse(scene)
    #print(params)

    opt = mi.ad.Adam(lr=0.001)

    # prepare latent parameters
    opt[f"light_pose_trans"]      = mi.Vector3f([0.001, 0, 0])
    # for isotropic lightsources, we only optimize 2DOF of rotations
    opt[f"light_pose_so3_rot_x"]  = mi.Float(0) 
    opt[f"light_pose_so3_rot_y"]  = mi.Float(0)

    params[light_pose_key] = exp_map_SO3XR3(opt[f"light_pose_trans"], mi.Vector3f(opt[f"light_pose_so3_rot_x"], opt[f"light_pose_so3_rot_y"], 0))
    
    params.update(opt)
    dr.set_grad_enabled(opt[f"light_pose_trans"], True)
    dr.set_grad_enabled(opt[f"light_pose_so3_rot_x"], True)
    dr.set_grad_enabled(opt[f"light_pose_so3_rot_y"], True)

    latest_board_trans= np.array(ref_poses[0].matrix)

    for it in range(50):
        for i, T_board2world in enumerate(ref_poses):

            # set board pose
            diff_pose = np.array(T_board2world.matrix).dot(np.linalg.inv(latest_board_trans))
            apply_transformation(vertex_key, params, diff_pose) 
            # save transformation to undo it in next iter
            latest_board_trans = np.array(T_board2world.matrix)
            params.update()
            
            # render
            img = mi.render(scene, params, sensor=0, spp=spp, seed=42)

            # compute error 
            err =  dr.sum (dr.sqr( img - ref_imgs[i] )) / len(img)
            mi.Log(mi.LogLevel.Info, f"Error it {it} view {i}: {err}")

            # backprop
            dr.backward(err)

            if False:
                mi.Log(mi.LogLevel.Debug,f"GRAD light pose trans {dr.grad(opt['light_pose_trans'])}")
                mi.Log(mi.LogLevel.Debug,f"GRAD light_pose_so3_rot_x {dr.grad(opt['light_pose_so3_rot_x'])}")
                mi.Log(mi.LogLevel.Debug,f"GRAD light_pose_so3_rot_y {dr.grad(opt['light_pose_so3_rot_y'])}")

            # step
            opt.step()
            

            params[light_pose_key] = exp_map_SO3XR3(opt[f"light_pose_trans"], mi.Vector3f(opt[f"light_pose_so3_rot_x"], opt[f"light_pose_so3_rot_y"], 0))


            if (dr.any(dr.isnan(params[light_pose_key]))):
                mi.Log(mi.LogLevel.Debug, "dr.any(dr.isnan( light_pose_trans] )): %s"    %dr.any(dr.isnan(opt[f"light_pose_trans"]) )) 
                mi.Log(mi.LogLevel.Debug, "dr.any(dr.isnan( light_pose_so3_rot_x )): %s" %dr.any(dr.isnan(opt[f"light_pose_so3_rot_x"])))
                mi.Log(mi.LogLevel.Debug, "dr.any(dr.isnan( light_pose_so3_rot_y )): %s" %dr.any(dr.isnan(opt[f"light_pose_so3_rot_y"])))
                
                raise Exception("Encountered NAN in pose gradients") 

            params.update()

            if False:
                mi.util.write_bitmap(f"./img_opt_view_it_{it}_view_{i}.png", img)



if __name__ == "__main__":

    #test_exp_log_SO3XR3()
    #exit(0)

    # use spotlight && pose estimation ==> works w/ llvm_ad_rgb, works w/cuda_ad_rgb 
    ltype = "spotlight" # ["spotlight", "tablight"]
    
    # obtain reference images
    ref_imgs, ref_poses = render(64, ltype)
    # optimize spotlight pose w/ 64 spp => works
    print ("++++++ Spotlight 64 SPP ++++++")
    optimize(64, ref_imgs, ref_poses, ltype)
    
    print ("++++++ Tablight 1 SPP ++++++")
    ltype = "tablight" # ["spotlight", "tablight"]
    ref_imgs, ref_poses = render(64, ltype)
    # optimize tablight pose  ==> works w/ llvm_ad_rgb and SPP = 1, works w/ cuda_ad_rgb and SPP = 1
    optimize(1, ref_imgs, ref_poses, ltype)
    print ("++++++ Tablight 64 SPP ++++++")
    # optimize tablight pose ==> fails w/ llvm_ad_rgb and SPP > 1, fails w/ cuda_ad_rgb and SPP > 1
    optimize(64, ref_imgs, ref_poses, ltype)



    