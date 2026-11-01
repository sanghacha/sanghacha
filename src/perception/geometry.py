"""Metric RGB-D geometry in robot_base; no simulator object IDs."""
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

def backproject(depth, k, transform):
    assert depth.ndim==2 and k.shape==(3,3) and transform.shape==(4,4)
    assert np.isfinite(k).all() and np.isfinite(transform).all() and k[0,0]>0 and k[1,1]>0
    assert np.allclose(transform[3],[0,0,0,1])
    assert np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-6)
    y,x=np.indices(depth.shape)
    points=np.stack([(x+.5-k[0,2])*depth/k[0,0],(y+.5-k[1,2])*depth/k[1,1],depth],axis=-1)
    return points@transform[:3,:3].T+transform[:3,3]

def cloud(points, colors=None):
    p=o3d.geometry.PointCloud()
    p.points=o3d.utility.Vector3dVector(np.asarray(points,dtype=np.float64))
    if colors is not None:p.colors=o3d.utility.Vector3dVector(np.asarray(colors,dtype=np.float64))
    return p

def clean_cloud(points, colors):
    p=cloud(points,colors).voxel_down_sample(.002)
    if len(p.points)>35:
        p,_=p.remove_statistical_outlier(nb_neighbors=16,std_ratio=2.5)
    return p

def describe(p):
    pts=np.asarray(p.points)
    assert len(pts)>=4 and np.isfinite(pts).all()
    # Open3D's robust OBB also supports partial/coplanar visible surfaces.
    covariance=np.cov(pts.T)
    eig=np.linalg.eigvalsh(covariance)
    try:
        if eig[0] < 1e-12:
            raise RuntimeError('Planar observed surface')
        box=p.get_oriented_bounding_box(robust=True)
        center=np.asarray(box.center);rotation=np.asarray(box.R);extent=np.asarray(box.extent)
    except RuntimeError:
        lo,hi=pts.min(0),pts.max(0)
        center=(lo+hi)/2;rotation=np.eye(3);extent=hi-lo
    return dict(visible_surface_centroid_m=np.median(pts,axis=0).tolist(),
                aabb_min_m=pts.min(0).tolist(),aabb_max_m=pts.max(0).tolist(),
                mean_rgb=np.asarray(p.colors).mean(0).tolist(),
                obb=dict(center_m=center.tolist(),extent_m=np.maximum(extent,1e-6).tolist(),
                         rotation=rotation.tolist(),quaternion_xyzw=Rotation.from_matrix(rotation).as_quat().tolist()),
                pose_kind='partial_surface_OBB_estimate_not_registered_6D_pose',
                rotation_observable=bool(eig[0]>1e-7 and eig[2]>1.3*eig[1] and eig[1]>1.3*eig[0]),
                points=len(pts))

def snapshot_tsdf(frames, output, voxel=.003):
    """New volume on EVERY snapshot. Moving objects never accumulate across time."""
    volume=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel,sdf_trunc=4*voxel,
              color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for f in frames.values():
        d=f['depth'].copy().astype(np.float32)
        xyz=backproject(d,f['K'],f['T'])
        valid=(d>0)&(d<2)&np.isfinite(d)&(xyz[:,:,0]>-.25)&(xyz[:,:,0]<.48)&(abs(xyz[:,:,1])<.34)&(xyz[:,:,2]>-.015)&(xyz[:,:,2]<.55)
        d[~valid]=0
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(f['rgb'].copy()),
             o3d.geometry.Image(d),depth_scale=1.,depth_trunc=2.,convert_rgb_to_intensity=False)
        h,w=d.shape;k=f['K']
        intr=o3d.camera.PinholeCameraIntrinsic(w,h,k[0,0],k[1,1],k[0,2],k[1,2])
        # Open3D integrate expects base/world -> camera, opposite to stored T.
        volume.integrate(rgbd,intr,np.linalg.inv(f['T']))
    p=volume.extract_point_cloud()
    assert len(p.points)>0,'Empty TSDF reconstruction'
    p.colors=o3d.utility.Vector3dVector(np.clip(np.asarray(p.colors),0,1))
    assert o3d.io.write_point_cloud(str(output),p)
    return len(p.points)
