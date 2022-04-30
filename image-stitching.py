import os
import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def homography_function(x, y, H): # x&y-> N-by-1, H-> 3-by-3, return-> (N-by-1,N-by-1)
  D = (H[2,0] * x + H[2,1] * y + 1)

  xs = (H[0,0] * x + H[0,1] * y + H[0,2]) / D
  ys = (H[1,0] * x + H[1,1] * y + H[1,2]) / D

  return xs, ys

def homography_jacobian(x, y, H): # x&y-> N-by-1, H-> 3-by-3, J-> 2N-by-8
  N = np.shape(x)[0]
  J = np.zeros(shape=(2*N, 8), dtype=float)

  D = (H[2,0] * x + H[2,1] * y + 1)

  xs = (H[0,0] * x + H[0,1] * y + H[0,2]) / D
  ys = (H[1,0] * x + H[1,1] * y + H[1,2]) / D

  J[0:N,0:1] = x / D
  J[0:N,1:2] = y / D
  J[0:N,2:3] = 1 / D
  J[0:N,6:7] = - x * xs
  J[0:N,7:8] = - y * ys

  J[N:2*N,3:4] = x / D
  J[N:2*N,4:5] = y / D
  J[N:2*N,5:6] = 1 / D
  J[N:2*N,6:7] = - x * xs
  J[N:2*N,7:8] = - y * ys

  return J   

def find_homography(x, y, xp, yp, iters): # x,y,xp&yp-> N-by-1
  # define H here
  H = np.zeros(shape=(3,3), dtype=float)
  H[2,2] = 1

  for _ in range(iters):
    # Compute homography coordinates
    xs, ys = homography_function(x, y, H)

    # Compute residuals
    rx, ry = xs - xp, ys - yp
    r = np.vstack((rx,ry))

    # Compute Jacobian
    J = homography_jacobian(x,y,H)

    # Compute update via linear least squares
    delta = np.linalg.pinv(J.transpose() @ J) @ J.transpose() @ r

    # Update homography matrix
    delta = np.resize(np.vstack((delta,0)), (3,3))
    H = H - delta

  return H

def get_match_coordinates(sift_keypoints1, sift_keypoints2, matches):
  N = len(matches)

  x = np.zeros(shape=(N, 1), dtype=float)
  y = np.zeros(shape=(N, 1), dtype=float)
  xp = np.zeros(shape=(N, 1), dtype=float)
  yp = np.zeros(shape=(N, 1), dtype=float)

  for k in range(N):
    index1, index2, _, _ = matches[k]

    x[k,0] = np.floor(sift_keypoints1[index1].pt[0])
    y[k,0] = np.floor(sift_keypoints1[index1].pt[1])
    xp[k,0] = np.floor(sift_keypoints2[index2].pt[0])
    yp[k,0] = np.floor(sift_keypoints2[index2].pt[1])

  return x,y,xp,yp

def get_random_inliers(x, y, xp, yp, n):
  inliers_idx = np.random.choice(np.arange(len(x)), n, replace=False)
  outliers_idx = np.ones(shape=len(x), dtype=bool)
  outliers_idx[inliers_idx] = 0
  outliers_idx = np.nonzero(outliers_idx)[0]

  inliers = (x[inliers_idx], y[inliers_idx], xp[inliers_idx], yp[inliers_idx])
  outliers = (x[outliers_idx], y[outliers_idx], xp[outliers_idx], yp[outliers_idx])
  
  return inliers, inliers_idx, outliers, outliers_idx

def get_random_inliers_with_index(x, y, xp, yp, idx, n):
  inliers_idx = np.random.choice(idx, n, replace=False)
  outliers_idx = np.ones(shape=len(x), dtype=bool)
  outliers_idx[inliers_idx] = 0
  outliers_idx = np.nonzero(outliers_idx)[0]

  inliers = (x[inliers_idx], y[inliers_idx], xp[inliers_idx], yp[inliers_idx])
  outliers = (x[outliers_idx], y[outliers_idx], xp[outliers_idx], yp[outliers_idx])
  
  return inliers, inliers_idx, outliers, outliers_idx

def ransac(sift_keypoints1, sift_keypoints2, matches, min_points, req_points, gn_iters, max_iters, ransac_threshold):
  H_best = None
  err_best = np.inf
  x_best, y_best, xp_best, yp_best, idx_best = None, None, None, None, None

  print('      Extracting matched feature point coordinates...')
  x, y, xp, yp = get_match_coordinates(sift_keypoints1, sift_keypoints2, matches)

  print('      Running RANSAC iterations...')
  for num_iter in range(max_iters):
    if not idx_best is None:
      # Get 'num_inliers/2' random inliers from the best set
      inliers, idx_inl, outliers, idx_oth = get_random_inliers_with_index(x, y, xp, yp, idx_best, int(idx_best.shape[0] / 2))
    else:
      # Get 'min_points' random inliers from the set
      inliers, idx_inl, outliers, idx_oth = get_random_inliers(x, y, xp, yp, min_points)

    x_inl, y_inl, xp_inl, yp_inl = inliers
    x_oth, y_oth, xp_oth, yp_oth = outliers

    # Fit a homography to randomly selected inliers
    H = find_homography(x_inl, y_inl, xp_inl, yp_inl, gn_iters)

    # Evaluate homography on the rest of the set
    xs_oth, ys_oth = homography_function(x_oth, y_oth, H)
    r_oth = np.sqrt(np.square(xs_oth - xp_oth) + np.square(ys_oth - yp_oth))

    # Add good fit points to inliers
    idx = np.where(r_oth < ransac_threshold)[0]

    if idx.shape[0] > 0:
      x_inl = np.concatenate((x_inl, x_oth[idx, :]))
      y_inl = np.concatenate((y_inl, y_oth[idx, :]))
      xp_inl = np.concatenate((xp_inl, xp_oth[idx, :]))
      yp_inl = np.concatenate((yp_inl, yp_oth[idx, :]))
      idx_inl = np.concatenate((idx_inl, idx_oth[idx]))

    # Check if found model has enough inliers
    if x_inl.shape[0] >= req_points:
      # Fit a homography again to all found inliers
      H = find_homography(x_inl, y_inl, xp_inl, yp_inl, gn_iters)

      # Evaluate homography on all found inliers
      xs_inl, ys_inl = homography_function(x_inl, y_inl, H)
      err = np.mean(np.sqrt(np.square(xs_inl - xp_inl) + np.square(ys_inl - yp_inl)))

      # Check if found error is better than the best model
      if err < err_best:
        # Update best homography, error and inlier points
        H_best = H
        err_best = err
        x_best, y_best, xp_best, yp_best, idx_best = x_inl, y_inl, xp_inl, yp_inl, idx_inl

  return H_best, x_best, y_best, xp_best, yp_best, idx_best

def match_feature_points(descriptors1, descriptors2, threshold):
  num_keypoints1, vec_dim = descriptors1.shape
  num_keypoints2, _ = descriptors2.shape

  matches = []

  for index1 in range(num_keypoints1):
    desc1 = descriptors1[index1,:]

    nearest_n1 = np.inf * np.ones(shape=(1, vec_dim), dtype=float)
    nearest_n1_index = np.inf
    nearest_n1_dist = np.inf

    nearest_n2 = np.inf * np.ones(shape=(1, vec_dim), dtype=float)
    nearest_n2_index = np.inf
    nearest_n2_dist = np.inf

    for index2 in range(num_keypoints2):
      desc2 = descriptors2[index2,:]

      temp_dist = np.linalg.norm(desc1 - desc2)

      if temp_dist < nearest_n1_dist:
        nearest_n2 = nearest_n1
        nearest_n2_index = nearest_n1_index
        nearest_n2_dist = nearest_n1_dist
        
        nearest_n1 = desc2
        nearest_n1_index = index2
        nearest_n1_dist = temp_dist
      elif temp_dist < nearest_n2_dist:
        nearest_n2 = desc2
        nearest_n2_index = index2
        nearest_n2_dist = temp_dist

    nndr = nearest_n1_dist / nearest_n2_dist
    if nndr < threshold:
      matches.append((index1, nearest_n1_index, nearest_n1_dist, nndr))
    
  return matches

def stitch(img1, img2, H, r_shift_prev, c_shift_prev, estimation_iters):
  img1_rows, img1_cols, _ = img1.shape
  img2_rows, img2_cols, _ = img2.shape

  img1_transformed_coordinates = np.zeros(shape=(img1_rows, img1_cols), dtype="i,i")
  r_min, c_min = np.inf, np.inf
  r_max, c_max = -np.inf, -np.inf

  # Transform image one
  for r in range(img1_rows):
    for c in range(img1_cols):
      xs, ys = c, r

      for H_i in reversed(H):
        xs, ys = homography_function(xs, ys, H_i)

      xs, ys = xs + c_shift_prev, ys + r_shift_prev
      xs, ys = int(xs), int(ys)

      if ys < r_min:
        r_min = ys

      if ys > r_max:
        r_max = ys

      if xs < c_min:
        c_min = xs

      if xs > c_max:
        c_max = xs

      img1_transformed_coordinates[r,c] = (ys,xs)

  # Calculate the size of the stitched image
  out_rows, out_cols = img2_rows, img2_cols

  if (r_min < 0):
    out_rows = out_rows - r_min

  if (r_max > img2_rows - 1):
    out_rows = out_rows + (r_max - img2_rows + 1)

  if (c_min < 0):
    out_cols = out_cols - c_min

  if (c_max > img2_cols - 1):
    out_cols = out_cols + (c_max - img2_cols + 1)

  out = np.zeros(shape=(out_rows, out_cols, 3), dtype=np.uint8)
  out_temp = np.zeros(shape=(out_rows, out_cols, 3), dtype=np.uint8)
  out_temp_map = np.zeros(shape=(out_rows, out_cols), dtype=bool)
  out_map1 = np.zeros(shape=(out_rows, out_cols), dtype=bool)
  out_map2 = np.zeros(shape=(out_rows, out_cols), dtype=bool)

  r_shift = 0
  if r_min < 0:
    r_shift = - r_min

  c_shift = 0
  if c_min < 0:
    c_shift = - c_min

  # Insert image one
  for r in range(img1_rows):
    for c in range(img1_cols):
      rt, ct = img1_transformed_coordinates[r,c]
      rt, ct = rt + r_shift, ct + c_shift

      out_temp[rt,ct,:] = img1[r,c,:]

      out_temp_map[rt,ct] = True

  # Estimate missing pixel values in image one
  for _ in range(estimation_iters):
    for r in range(1,out_rows-1):
      for c in range(1,out_cols-1):
        patch = out_temp[r-1:r+2,c-1:c+2,:]
        patch_map = out_temp_map[r-1:r+2,c-1:c+2]

        if not out_temp_map[r,c] and not np.all(patch_map == False):
          out_temp[r,c] = np.median(patch[patch_map], axis=0)
          out_map1[r,c] = True

    out_temp_map = np.logical_or(out_temp_map, out_map1)

  out_map1 = out_temp_map

  # Insert image two
  for r in range(img2_rows):
    for c in range(img2_cols):
      rt, ct = r + r_shift, c + c_shift

      if not np.all(img2[r,c,:] == 0):
        out[rt,ct,:] = img2[r,c,:]
        out_map2[rt,ct] = True

  # Merge two maps
  for r in range(out_rows):
    for c in range(out_cols):
      if out_map1[r,c]:
        if out_map2[r,c]:
          out[r,c,:] = ((out[r,c,:].astype(float) + out_temp[r,c,:].astype(float)) / 2).astype(np.uint8)
        else:
          out[r,c,:] = out_temp[r,c,:]
        
  return out, r_shift_prev + r_shift, c_shift_prev + c_shift

# ***** DEBUG *****
'''
def matches_to_opencv_matches(sift_descriptors1, sift_descriptors2, matches):
  opencv_matches = []
  for k in range(len(matches)):
    index1, index2, _, _ = matches[k]

    opencv_matches.append([cv.DMatch(index1,index2,np.linalg.norm(sift_descriptors1[index1,:] - sift_descriptors2[index2,:]))])

  return opencv_matches
'''
# ***** DEBUG *****

def image_stitching(image_paths, out_file, nndr_threshold, min_points, req_points, gn_iters, max_iters, ransac_threshold, estimation_iters):
  images = []
  sift_keypoints = []
  sift_descriptors = []
  
  for image_path in image_paths:
    # Read image parts
    image = cv.imread(image_path)

    # Create a SIFT instance
    sift = cv.SIFT_create()

    # Detect SIFT points 
    sift_keypoints_i, sift_descriptors_i = sift.detectAndCompute(image,None)

    images.append(image)
    sift_keypoints.append(sift_keypoints_i)
    sift_descriptors.append(sift_descriptors_i)

  out = images[0]
  H = []
  r_shift = 0
  c_shift = 0

  for i in range(len(images)-1):
    print('    Finding feature match points... [' + str(i) + ']')
    matches = match_feature_points(sift_descriptors[i+1], sift_descriptors[i], 0.8)

    # ***** DEBUG *****
    """
    matches = match_feature_points(sift_descriptors[i+1], sift_descriptors[i], 1)
    matches = matches_to_opencv_matches(sift_descriptors[i+1], sift_descriptors[i], matches)
    img_matches = cv.drawMatchesKnn(images[i+1], sift_keypoints[i+1], images[i], sift_keypoints[i], matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite('out/matches10.jpg',img_matches)
    
    matches = match_feature_points(sift_descriptors[i+1], sift_descriptors[i], 0.8)
    matches = matches_to_opencv_matches(sift_descriptors[i+1], sift_descriptors[i], matches)
    img_matches = cv.drawMatchesKnn(images[i+1], sift_keypoints[i+1], images[i], sift_keypoints[i], matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite('out/matches08.jpg',img_matches)

    matches = match_feature_points(sift_descriptors[i+1], sift_descriptors[i], 0.8)
    H_i, x, y, xp, yp, idx = ransac(sift_keypoints[i+1], sift_keypoints[i], matches, min_points=min_points, req_points=req_points, gn_iters=gn_iters, max_iters=max_iters, ransac_threshold=ransac_threshold)
    matches = [matches[i] for i in idx]
    matches = matches_to_opencv_matches(sift_descriptors[i+1], sift_descriptors[i], matches)
    img_matches = cv.drawMatchesKnn(images[i+1], sift_keypoints[i+1], images[i], sift_keypoints[i], matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite('out/matches08_ransac.jpg',img_matches)
    """
    # ***** DEBUG *****

    print('    Running RANSAC algorithm to fit a homography on matched feature points... [' + str(i) + ']')
    H_i, x, y, xp, yp, idx = ransac(sift_keypoints[i+1], sift_keypoints[i], matches, min_points=min_points, req_points=req_points, gn_iters=gn_iters, max_iters=max_iters, ransac_threshold=ransac_threshold)

    H.append(H_i)

    print('    Stitcing images... [' + str(i) + ']')
    out, r_shift, c_shift = stitch(images[i+1], out, H, r_shift, c_shift, estimation_iters=estimation_iters)

  cv.imwrite(out_file, out)
  
  out_disp = cv.cvtColor(out, cv.COLOR_BGR2RGB)
  plt.imshow(out_disp)
  plt.show()

def main():
  image_stitching(sys.argv[1:], 'stitched_image.jpg', nndr_threshold=0.8, min_points=10, req_points=20, gn_iters=100, max_iters=1000, ransac_threshold=3, estimation_iters=1)

if __name__ == "__main__":
    main()
