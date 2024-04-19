import drjit as dr
import mitsuba as mi
#### SO3 X R3 pose optimization ####

def scalar_exp_map_SO3(log_rotation, eta=1e-4):
    """Exponential map so3 -> SO3 
    Inspired by Nerfstudio, which in turn was inspired by Pytorch3D

    Args:
        log_rotation (mi.Vector3f): _description_
        eta (float, optional): numerical padding. Defaults to 1e-4.

    Returns:
        mi.Matrix3f: 3X3 Matrix representing the SO3 rotation
    """
    nrms = dr.sum(log_rotation*log_rotation)

    rot_angles = dr.sqrt(dr.clamp(nrms, eta, 2*dr.pi))
    rot_angles_inv = 1. / rot_angles
    fac1 = rot_angles_inv * dr.sin(rot_angles)
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - dr.cos(rot_angles))
    

    skews = mi.ScalarMatrix3f([[0,               -log_rotation[2], log_rotation[1]],
                               [log_rotation[2],  0,              -log_rotation[0]],
                               [-log_rotation[1], log_rotation[0], 0]])

    skews_square = skews @ skews

    id = mi.ScalarMatrix3f( 1,0,0,
                      0,1,0,
                      0,0,1 )
    
    res = mi.ScalarMatrix3f( 1,0,0,
                       0,1,0,
                       0,0,1,
                     )


    # TODO: no slicing on non-dynamic arrays, nicer solution?
    # Compose rotation 
    res[0,0] = fac1 * skews[0,0] +  fac2 * skews_square[0,0] + id[0,0]
    res[0,1] = fac1 * skews[0,1] +  fac2 * skews_square[0,1] + id[0,1]
    res[0,2] = fac1 * skews[0,2] +  fac2 * skews_square[0,2] + id[0,2]

    res[1,0] = fac1 * skews[1,0] +  fac2 * skews_square[1,0] + id[1,0]
    res[1,1] = fac1 * skews[1,1] +  fac2 * skews_square[1,1] + id[1,1]
    res[1,2] = fac1 * skews[1,2] +  fac2 * skews_square[1,2] + id[1,2]

    res[2,0] = fac1 * skews[2,0] +  fac2 * skews_square[2,0] + id[2,0]
    res[2,1] = fac1 * skews[2,1] +  fac2 * skews_square[2,1] + id[2,1]
    res[2,2] = fac1 * skews[2,2] +  fac2 * skews_square[2,2] + id[2,2]

    return res


def scalar_exp_map_SO3XR3(log_translation, log_rotation, eta=1e-4):
    """
    Compute exponential map of SO3XR3

    Inspired by Nerfstudio, which in turn was inspired by pyTorch3D

    Args:
        log_translation: 3d vector tangent vector with translational values
        log_rotation: 3d vector tangent vector with rotational values
    
    Returns:
        mi.Transform 4f R T
                        0 1   
    """
    res = mi.ScalarMatrix4f( 1,0,0,0,
                       0,1,0,0,
                       0,0,1,0,
                       0,0,0,1
                     )
    
    # SO3 exp map
    SO3 = scalar_exp_map_SO3(log_rotation)

    res[0,0] = SO3[0,0]
    res[0,1] = SO3[0,1]
    res[0,2] = SO3[0,2]
    res[1,0] = SO3[1,0]
    res[1,1] = SO3[1,1]
    res[1,2] = SO3[1,2]
    res[2,0] = SO3[2,0]
    res[2,1] = SO3[2,1]
    res[2,2] = SO3[2,2]
   

    # directly use R3
    res[0,3] = log_translation[0]
    res[1,3] = log_translation[1]
    res[2,3] = log_translation[2]
    
    return mi.ScalarTransform4f(res)


def exp_map_SO3(log_rotation, eta=1e-4):
    """Exponential map so3 -> SO3 
    Inspired by Nerfstudio, which in turn was inspired by Pytorch3D

    Args:
        log_rotation (mi.Vector3f): _description_
        eta (float, optional): numerical padding. Defaults to 1e-4.

    Returns:
        mi.Matrix3f: 3X3 Matrix representing the SO3 rotation
    """
    nrms = dr.sum(log_rotation*log_rotation)

    rot_angles = dr.sqrt(dr.clamp(nrms, eta, 2*dr.pi))
    rot_angles_inv = 1. / rot_angles
    fac1 = rot_angles_inv * dr.sin(rot_angles)
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - dr.cos(rot_angles))
    

    skews = mi.Matrix3f([[0,               -log_rotation[2], log_rotation[1]],
                         [log_rotation[2],  0,              -log_rotation[0]],
                         [-log_rotation[1], log_rotation[0], 0]])

    skews_square = skews @ skews

    id = mi.Matrix3f( 1,0,0,
                      0,1,0,
                      0,0,1 )
    
    res = mi.Matrix3f( 1,0,0,
                       0,1,0,
                       0,0,1 )
    

    # TODO: no slicing on non-dynamic arrays, nicer solution?
    # Compose rotation 
    res[0,0] = fac1 * skews[0,0] +  fac2 * skews_square[0,0] + id[0,0]
    res[0,1] = fac1 * skews[0,1] +  fac2 * skews_square[0,1] + id[0,1]
    res[0,2] = fac1 * skews[0,2] +  fac2 * skews_square[0,2] + id[0,2]

    res[1,0] = fac1 * skews[1,0] +  fac2 * skews_square[1,0] + id[1,0]
    res[1,1] = fac1 * skews[1,1] +  fac2 * skews_square[1,1] + id[1,1]
    res[1,2] = fac1 * skews[1,2] +  fac2 * skews_square[1,2] + id[1,2]

    res[2,0] = fac1 * skews[2,0] +  fac2 * skews_square[2,0] + id[2,0]
    res[2,1] = fac1 * skews[2,1] +  fac2 * skews_square[2,1] + id[2,1]
    res[2,2] = fac1 * skews[2,2] +  fac2 * skews_square[2,2] + id[2,2]

    return res


def exp_map_SO3XR3(log_translation, log_rotation, eta=1e-4):
    """
    Compute exponential map of SO3XR3

    Inspired by Nerfstudio, which in turn was inspired by pyTorch3D

    Args:
        log_translation: 3d vector tangent vector with translational values
        log_rotation: 3d vector tangent vector with rotational values
    
    Returns:
        mi.Transform 4f R T
                        0 1   
    """
    res = mi.Matrix4f( 1,0,0,0,
                       0,1,0,0,
                       0,0,1,0,
                       0,0,0,1 )
    
    # SO3 exp map
    SO3 = exp_map_SO3(log_rotation)

    res[0,0] = SO3[0,0]
    res[0,1] = SO3[0,1]
    res[0,2] = SO3[0,2]
    res[1,0] = SO3[1,0]
    res[1,1] = SO3[1,1]
    res[1,2] = SO3[1,2]
    res[2,0] = SO3[2,0]
    res[2,1] = SO3[2,1]
    res[2,2] = SO3[2,2]
   

    # directly use R3
    res[0,3] = log_translation[0]
    res[1,3] = log_translation[1]
    res[2,3] = log_translation[2]
    
    return mi.Transform4f(res)


def scalar_log_map_SO3(x, eta=1e-8):
    """Log map SO3 -> so3

    from Integrating Generic Sensor Fusion Algorithms with
    Sound State Representations through Encapsulation of Manifolds
    Christoph Hertzberg, Rene Wagnerc, Udo Frese, Lutz Schroeder
    https://arxiv.org/pdf/1107.1119.pdf
    Eq 27

    Args:
        x (mi.Matrix3f): SO3 rotation
        eta (float, optional): numerical padding. Defaults to 1e-8.

    Returns:
        mi.Vector3f: 3X1 so3 rotation
    """
   
    
    angle = (dr.trace(x) -1) / 2
    theta = dr.acos(dr.clamp(angle,-1 + eta, 1 - eta))

    return (theta / (2* dr.sin(theta) ) ) *  mi.ScalarVector3f( x[2,1] - x[1,2] , x[0,2] - x[2,0], x[1,0] - x[0,1] )


def scalar_log_map_SO3XR3(SO3XR3, eta=1e-8):
    """Compound log of SO3XR3, translation will stay untouched

    Args:
        SO3XR3 (mi.Transform4f): SO3XR3 transformation
        eta (float, optional): numerical padding. Defaults to 1e-8.

    Returns:
        mi.Vector3f: translation
        mi.Vector3f: log rotation
    """
    
    
    trans = mi.ScalarVector3f(SO3XR3.matrix[0,3], SO3XR3.matrix[1,3], SO3XR3.matrix[2,3])

    SO3 = mi.ScalarMatrix3f([ [SO3XR3.matrix[0, 0], SO3XR3.matrix[0, 1], SO3XR3.matrix[0, 2]],
                        [SO3XR3.matrix[1, 0], SO3XR3.matrix[1, 1], SO3XR3.matrix[1, 2]],
                        [SO3XR3.matrix[2, 0], SO3XR3.matrix[2, 1], SO3XR3.matrix[2, 2]]] )
    
    so3_rot   = scalar_log_map_SO3(SO3, eta)
    
    return trans, so3_rot


def test_exp_log_SO3XR3():

    scalar_log_trans = mi.ScalarVector3f([1,2,3]) 
    scalar_log_rot = mi.ScalarVector3f([dr.pi/2, dr.pi/4, 0]) 

    log_trans = mi.Vector3f([1,2,3]) 
    log_rot = mi.Vector3f([dr.pi/2, dr.pi/4, 0]) 

    print("Initial log_trans", log_trans)
    print("Initial log_rot", log_rot) 

    scalar_SO3XR3 = scalar_exp_map_SO3XR3(scalar_log_trans, scalar_log_rot)
    SO3XR3 = exp_map_SO3XR3(log_trans, log_rot)

    print("SO3XR3", SO3XR3)
    print("scalar_SO3XR3", scalar_SO3XR3)
    

    _log_trans, _log_rot = scalar_log_map_SO3XR3(scalar_SO3XR3)

    print("Final log_trans", _log_trans)
    print("Final log_rot", _log_rot) 

#### SO3 X R3 pose optimization ####

def get_scene(spp, light_type, T_light2world, T_board2world,
              spot_params=[[1,2,3],15,45],
              tabspot_params=[[1,2,3], "0.0,1., 2.5,0.999, 5.0,0.998, 7.5,0.95, 10.0,0.96, 15.0,0.915, 20.0,0.87, 25.0,0.81, 30.0,0.75, 35.0,0.63, 40.0,0.5, 45.0,0.28, 50.0,0.2, 55.0,0.16, 60.0,0., 65.0,0., 90.0,0.0"]):


    scene_dict = dict({"type": "scene"}) # this is a scence, defined by nested dictionaries

    # add integrator
    scene_dict.update( {
                "myintegrator": {
                    "type"          : "prbvolpath",
                    "max_depth"     : 16, # max path depth
                    "rr_depth"      : 32,  # depth after wich russian roulette is started
                    "hide_emitters" : False,                      # Hide directly visible emitters.
                },
        } )

    # construct film
    mi.Log(mi.LogLevel.Info, "Using wideband film for sensor")
    film = {    "type" : "hdrfilm",
                "rfilter" : {   "type" : "gaussian"},
                                "width" :  128,
                                "height" : 128,
                                "pixel_format": "rgb",
                                "component_format": "float32"
            }
            

    # add camera
    mi.Log(mi.LogLevel.Info, "Adding perspective sensor")
    scene_dict.update( {"mysensor" : {
                                    "type"  : "perspective",
                                    "id"    : "perspective_camera",
                                    "fov"   : 70,
                                    "near_clip": 0.00001,
                                    "far_clip": 200,
                                    "to_world" : mi.ScalarTransform4f(),
                                    "myfilm" : film,
                                    "mysampler" : {
                                        "type" : "independent", # "stratified",  # "multijitter", # "independent",
                                        "sample_count" : spp,
                                        }, 
                                    },
                        } )


    # add light
    if light_type == "spotlight":
        mi.Log(mi.LogLevel.Info, "Adding spot /w pose\n%s"% (T_light2world))
        scene_dict.update({ "spot"  : {  "type"        : "spot",
                                         "to_world"      : T_light2world,
                                         "intensity"     : {
                                                            "type" : "rgb",
                                                            "value": spot_params[0]
                                                            },
                                         "beam_width"    : spot_params[1],
                                         "cutoff_angle"  : spot_params[2],
                                    }
                                            
                            })


    elif light_type == "tablight":
        mi.Log(mi.LogLevel.Info, "Adding tabspot /w pose\n%s"% (T_light2world))
        scene_dict.update({ "tab_spot" : {   "type"        : "tabspot",
                                                            "to_world"      : T_light2world,
                                                            "intensity"     : {
                                                                                "type" : "rgb",
                                                                                "value": tabspot_params[0],
                                                                                },
                                                            "falloff_table": tabspot_params[1],
                                         }
                                            
                            })


    # add white plane
    if True:
        mi.Log(mi.LogLevel.Info, "Adding board /w pose\n%s"% (T_board2world))
        scene_dict.update( { "board": { "type" : "obj",
                                        "to_world" : T_board2world,
                                       "flip_normals": True,
                                        "filename" : "./resources/data/common/meshes/rectangle.obj",
                                        "material": {
                                                        "type": "diffuse",
                                                        "reflectance": {  "type": "rgb",
                                                                          "value": [1,1,1] }
                                                                        },
    
                                        }
        })

    return mi.load_dict(scene_dict)


# Apply the transformation to mesh vertex position and update scene (e.g. Optix BVH)
def apply_transformation(obj_vertex_pos_key, params, new_pose):

    trafo =  mi.Transform4f( new_pose )
    initial_vertex_positions = dr.unravel(mi.Point3f, params[obj_vertex_pos_key])
    params[obj_vertex_pos_key] = dr.ravel(trafo @ initial_vertex_positions)
    params.update()