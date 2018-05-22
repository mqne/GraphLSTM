import numpy as np
import cv2


def get_matrix_2D(center, rot, trans, scale):
    ca = np.cos(rot)
    sa = np.sin(rot)
    sc = scale
    cx = center[0]
    cy = center[1]
    tx = trans[0]
    ty = trans[1]
    t = np.array([[ca * sc, -sa * sc, sc * (ca * (-tx - cx) + sa * ( cy + ty)) + cx],
                  [sa * sc,  ca * sc, sc * (ca * (-ty - cy) + sa * (-tx - cx)) + cy]])
    return t


def get_matrix_2D_3D(center, rot, trans, scale):
    mat2D = get_matrix_2D(center, rot, trans, scale)
    mat2D_extended = np.hstack([mat2D[:,:2], [[0],[0]], mat2D[:,2].reshape([2,1])])
    mat3D = np.eye(4)
    mat3D[:2] = mat2D_extended
    return mat2D, mat3D


def transform_image_and_points(image, points, center, rot, trans, trans_d, scale):
    mat2D, mat3D = get_matrix_2D_3D(center, rot, trans, scale)
    
    out_image = None
    if image is not None:
        assert(type(image) == np.ndarray and len(image.shape) == 2)
        out_image = cv2.warpAffine(image, mat2D.reshape([2,3]), image.shape, flags=cv2.INTER_NEAREST)
        out_image = np.clip(out_image + trans_d, 0, np.inf)
        
    out_points = None
    if points is not None:
        assert(type(points) == np.ndarray
               and len(points.shape) == 2
               and points.shape[1] == 3)
        out_points = np.zeros([points.shape[0], 3])
        for i, pt in enumerate(points):
            out_pt = np.dot(mat3D, [pt[0], pt[1], pt[2], 1.])
            out_pt = out_pt[:3] / out_pt[3]
            out_pt[2] += trans_d
            out_points[i] = out_pt
        
    return out_image, out_points
