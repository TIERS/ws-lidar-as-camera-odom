#!/usr/bin/env python
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2018
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Daniel DeTone (ddetone)
#                       Tomasz Malisiewicz (tmalisiewicz)
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import argparse
import glob
import numpy as np
import os
import time
import cv2
import torch
from sklearn.neighbors import NearestNeighbors
import csv

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')

# Jet colormap for visualization.
myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])

class SuperPointNet(torch.nn.Module):
  """ Pytorch definition of SuperPoint Network. """
  def __init__(self):
    super(SuperPointNet, self).__init__()
    self.relu = torch.nn.ReLU(inplace=True)
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
    # Shared Encoder.
    self.conv1a = torch.nn.Conv2d(1,  c1, kernel_size=3, stride=1, padding=1)
    self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
    self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
    self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
    self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
    self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
    self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
    self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
    # Detector Head.
    self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
    # Descriptor Head.
    self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
    # Shared Encoder.
    x = self.relu(self.conv1a(x))
    x = self.relu(self.conv1b(x))
    x = self.pool(x)
    x = self.relu(self.conv2a(x))
    x = self.relu(self.conv2b(x))
    x = self.pool(x)
    x = self.relu(self.conv3a(x))
    x = self.relu(self.conv3b(x))
    x = self.pool(x)
    x = self.relu(self.conv4a(x))
    x = self.relu(self.conv4b(x))
    # Detector Head.
    cPa = self.relu(self.convPa(x))
    semi = self.convPb(cPa)
    # Descriptor Head.
    cDa = self.relu(self.convDa(x))
    desc = self.convDb(cDa)
    dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
    desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    return semi, desc

class SuperPointFrontend(object):
  """ Wrapper around pytorch net to help with pre and post image processing. """
  def __init__(self, weights_path, nms_dist, conf_thresh, nn_thresh,
               cuda=False):
    self.name = 'SuperPoint'
    self.cuda = cuda
    self.nms_dist = nms_dist
    self.conf_thresh = conf_thresh
    self.nn_thresh = nn_thresh # L2 descriptor distance for good match.
    self.cell = 8 # Size of each output cell. Keep this fixed.
    self.border_remove = 4 # Remove points this close to the border.

    # Load the network in inference mode.
    self.net = SuperPointNet()
    if cuda:
      # Train on GPU, deploy on GPU.
      self.net.load_state_dict(torch.load(weights_path))
      self.net = self.net.cuda()
    else:
      # Train on GPU, deploy on CPU.
      self.net.load_state_dict(torch.load(weights_path,
                               map_location=lambda storage, loc: storage))
    self.net.eval()

  def nms_fast(self, in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T
  
    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
  
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).
  
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
  
    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int) # Track NMS data.
    inds = np.zeros((H, W)).astype(int) # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2,:])
    corners = in_corners[:,inds1]
    rcorners = corners[:2,:].round().astype(int) # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
      return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
      out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
      return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
      grid[rcorners[1,i], rcorners[0,i]] = 1
      inds[rcorners[1,i], rcorners[0,i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
      # Account for top and left padding.
      pt = (rc[0]+pad, rc[1]+pad)
      if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
        grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
        grid[pt[1], pt[0]] = -1
        count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid==-1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

  def run(self, img):
    """ Process a numpy image to extract points and descriptors.
    Input
      img - HxW numpy float32 input image in range [0,1].
    Output
      corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      desc - 256xN numpy array of corresponding unit normalized descriptors.
      heatmap - HxW numpy heatmap in range [0,1] of point confidences.
      """
    assert img.ndim == 2, 'Image must be grayscale.'
    assert img.dtype == np.float32, 'Image must be float32.'
    H, W = img.shape[0], img.shape[1]
    inp = img.copy()
    inp = (inp.reshape(1, H, W))
    inp = torch.from_numpy(inp)
    inp = torch.autograd.Variable(inp).view(1, 1, H, W)
    if self.cuda:
      inp = inp.cuda()
    # Forward pass of network.
    outs = self.net.forward(inp)
    semi, coarse_desc = outs[0], outs[1]
    # Convert pytorch -> numpy.
    semi = semi.data.cpu().numpy().squeeze()
    # --- Process points.
    dense = np.exp(semi) # Softmax.
    dense = dense / (np.sum(dense, axis=0)+.00001) # Should sum to 1.
    # Remove dustbin.
    nodust = dense[:-1, :, :]
    # Reshape to get full resolution heatmap.
    Hc = int(H / self.cell)
    Wc = int(W / self.cell)
    nodust = nodust.transpose(1, 2, 0)
    heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
    heatmap = np.transpose(heatmap, [0, 2, 1, 3])
    heatmap = np.reshape(heatmap, [Hc*self.cell, Wc*self.cell])
    xs, ys = np.where(heatmap >= self.conf_thresh) # Confidence threshold.
    if len(xs) == 0:
      return np.zeros((3, 0)), None, None
    pts = np.zeros((3, len(xs))) # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist) # Apply NMS.
    inds = np.argsort(pts[2,:])
    pts = pts[:,inds[::-1]] # Sort by confidence.
    # Remove points along border.
    bord = self.border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W-bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H-bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    # --- Process descriptor.
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
      desc = np.zeros((D, 0))
    else:
      # Interpolate into descriptor map using 2D point locations.
      samp_pts = torch.from_numpy(pts[:2, :].copy())
      samp_pts[0, :] = (samp_pts[0, :] / (float(W)/2.)) - 1.
      samp_pts[1, :] = (samp_pts[1, :] / (float(H)/2.)) - 1.
      samp_pts = samp_pts.transpose(0, 1).contiguous()
      samp_pts = samp_pts.view(1, 1, -1, 2)
      samp_pts = samp_pts.float()
      if self.cuda:
        samp_pts = samp_pts.cuda()
      desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
      desc = desc.data.cpu().numpy().reshape(D, -1)
      desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
    return pts, desc, heatmap

class PointTracker(object):
  """ Class to manage a fixed memory of points and descriptors that enables
  sparse optical flow point tracking.

  Internally, the tracker stores a 'tracks' matrix sized M x (2+L), of M
  tracks with maximum length L, where each row corresponds to:
  row_m = [track_id_m, avg_desc_score_m, point_id_0_m, ..., point_id_L-1_m].
  """

  def __init__(self, max_length, nn_thresh):
    if max_length < 2:
      raise ValueError('max_length must be greater than or equal to 2.')
    self.maxl = max_length
    self.nn_thresh = nn_thresh
    self.all_pts = []
    for n in range(self.maxl):
      self.all_pts.append(np.zeros((2, 0)))
    self.tracks = np.zeros((0, self.maxl+2))
    self.track_count = 0
    self.max_score = 9999

    self.last_desc = None
    self.last_pts  = None


    self.time2 = None
    self.evaluation_4 = None
    self.evaluation_5 = None
    self.evaluation_6 = None


####################################
  def match_score(self, pts1, pts2, matches):
    """
    desc1 is the last descriptor in the queue
    desc2 is the new coming descriptors
    pts1  is the last keypoints in the queue
    pts2  is the new coming keypoints
    """

    kPtsSource = []
    for i in range(pts1.shape[1]):
        x, y, _ = pts1[:, i]
        keypoint = cv2.KeyPoint(x, y, 1)  # Assuming size 1 for all keypoints
        kPtsSource.append(keypoint)

    kPtsRef = []
    for i in range(pts2.shape[1]):
        x, y, _ = pts2[:, i]
        keypoint = cv2.KeyPoint(x, y, 1)  # Assuming size 1 for all keypoints
        kPtsRef.append(keypoint)

    # Method3:
    src_points = []
    dst_points = []

    for match in matches.T:
        src_points.append(kPtsSource[int(match[0])].pt)
        dst_points.append(kPtsRef[int(match[1])].pt)

    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    _, inlier_mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 3)

    num_inliers = np.count_nonzero(inlier_mask)
    self.evaluation_5 = num_inliers / matches.shape[1] # matching_score

########################################
  def Distinctiveness(self, desc1, desc2, nn_ratio):
    # desc2 is the new coming descriptors
    # desc1 is the last descriptor in the queue
    """
    Performs kNN matching with k=2, finds the best two matches, and then filters
    good matches based on the provided distance ratio.

    Inputs:
      desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      nn_ratio - Distance ratio to filter good matches.

    Returns:
      matches - 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
        return np.zeros((3, 0))

    if nn_ratio <= 0.0:
        raise ValueError('\'nn_ratio\' should be positive')

    # Perform kNN matching with k=2
    # for each descriptor in desc2, we find 2 nearest neighbors from desc1.
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto', metric='euclidean').fit(desc1.T)
    distances, indices = nbrs.kneighbors(desc2.T)

    # Calculate the distance ratio
    ratio_mask = distances[:, 0] / distances[:, 1] < nn_ratio

    # Get the surviving point indices

    m_idx1 = indices[:, 0][ratio_mask]
    m_idx2 = np.arange(desc2.shape[1])[ratio_mask]
    scores = distances[:, 0][ratio_mask]

    # Populate the final 3xN match data structure
    matches = np.zeros((3, int(ratio_mask.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores

    self.evaluation_6 = (matches.shape[1]) / (desc2.shape[1])

  def nn_match_two_way(self, desc1, desc2, nn_thresh):

    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.

    Inputs:
      desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      nn_thresh - Optional descriptor distance below which is a good match.

    Returns:
      matches - 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
      return np.zeros((3, 0))
    if nn_thresh < 0.0:
      raise ValueError('\'nn_thresh\' should be non-negative')
    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches

  def get_offsets(self):
    """ Iterate through list of points and accumulate an offset value. Used to
    index the global point IDs into the list of points.

    Returns
      offsets - N length array with integer offset locations.
    """
    # Compute id offsets.
    offsets = []
    offsets.append(0)
    for i in range(len(self.all_pts)-1): # Skip last camera size, not needed.
      offsets.append(self.all_pts[i].shape[1])
    offsets = np.array(offsets)
    offsets = np.cumsum(offsets)
    return offsets

  def update(self, pts, desc):
    """ Add a new set of point and descriptor observations to the tracker.

    Inputs
      pts - 3xN numpy array of 2D point observations.
      desc - DxN numpy array of corresponding D dimensional descriptors.
    """
    if pts is None or desc is None:
      print('PointTracker: Warning, no points were added to tracker.')
      return
    assert pts.shape[1] == desc.shape[1]
    # Initialize last_desc.
    if self.last_desc is None:
      self.last_desc = np.zeros((desc.shape[0], 0))
    # Remove oldest points, store its size to update ids later.
    remove_size = self.all_pts[0].shape[1]
    self.all_pts.pop(0)
    self.all_pts.append(pts)
    # Remove oldest point in track.
    self.tracks = np.delete(self.tracks, 2, axis=1)
    # Update track offsets.
    for i in range(2, self.tracks.shape[1]):
      self.tracks[:, i] -= remove_size
    self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
    offsets = self.get_offsets()
    # Add a new -1 column.
    self.tracks = np.hstack((self.tracks, -1*np.ones((self.tracks.shape[0], 1))))
    # Try to append to existing tracks.
    matched = np.zeros((pts.shape[1])).astype(bool)
    t2 = time.time()
    matches = self.nn_match_two_way(self.last_desc, desc, self.nn_thresh)
    t3 = time.time()

    self.evaluation_4 = (matches.shape[1])/(pts.shape[1])
    self.time2 = (t3 - t2)*1000

    if self.last_pts is not None:
      self.match_score(self.last_pts, pts, matches)

    for match in matches.T:
      # Add a new point to it's matched track.
      id1 = int(match[0]) + offsets[-2]
      id2 = int(match[1]) + offsets[-1]
      found = np.argwhere(self.tracks[:, -2] == id1)
      if found.shape[0] > 0:
        matched[int(match[1])] = True
        row = int(found)
        self.tracks[row, -1] = id2
        if self.tracks[row, 1] == self.max_score:
          # Initialize track score.
          self.tracks[row, 1] = match[2]
        else:
          # Update track score with running average.
          # NOTE(dd): this running average can contain scores from old matches
          #           not contained in last max_length track points.
          track_len = (self.tracks[row, 2:] != -1).sum() - 1.
          frac = 1. / float(track_len)
          self.tracks[row, 1] = (1.-frac)*self.tracks[row, 1] + frac*match[2]
    # Add unmatched tracks.
    new_ids = np.arange(pts.shape[1]) + offsets[-1]
    new_ids = new_ids[~matched]
    new_tracks = -1*np.ones((new_ids.shape[0], self.maxl + 2))
    new_tracks[:, -1] = new_ids
    new_num = new_ids.shape[0]
    new_trackids = self.track_count + np.arange(new_num)
    new_tracks[:, 0] = new_trackids
    new_tracks[:, 1] = self.max_score*np.ones(new_ids.shape[0])
    self.tracks = np.vstack((self.tracks, new_tracks))
    self.track_count += new_num # Update the track count.
    # Remove empty tracks.
    keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
    self.tracks = self.tracks[keep_rows, :]

   ####################################################################################################################
    self.Distinctiveness(self.last_desc, desc, 0.8)

    # Store the last descriptors and points.
    self.last_desc = desc.copy()
    self.last_pts = pts.copy()
   ####################################################################################################################
    return self.time2, self.evaluation_4, self.evaluation_5, self.evaluation_6
  
  def get_tracks(self, min_length):
    """ 
    Retrieve point tracks of a given minimum length.
    Input
      min_length - integer >= 1 with minimum track length
    Output
      returned_tracks - M x (2+L) sized matrix storing track indices, where
        M is the number of tracks and L is the maximum track length.
    """
    if min_length < 1:
      raise ValueError('\'min_length\' too small.')
    valid = np.ones((self.tracks.shape[0])).astype(bool)
    good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
    # Remove tracks which do not have an observation in most recent frame.
    not_headless = (self.tracks[:, -1] != -1)
    keepers = np.logical_and.reduce((valid, good_len, not_headless))
    returned_tracks = self.tracks[keepers, :].copy()
    return returned_tracks

  def draw_tracks(self, out, tracks):
    """ 
    Visualize tracks all overlayed on a single image.
    Inputs
      out - numpy uint8 image sized HxWx3 upon which tracks are overlayed.
      tracks - M x (2+L) sized matrix storing track info.
    """
    # Store the number of points per camera.
    pts_mem = self.all_pts
    N = len(pts_mem) # Number of cameras/images.
    # Get offset ids needed to reference into pts_mem.
    offsets = self.get_offsets()
    # Width of track and point circles to be drawn.
    stroke = 1
    # Iterate through each track and draw it.
    for track in tracks:
      clr = myjet[int(np.clip(np.floor(track[1]*10), 0, 9)), :]*255
      for i in range(N-1):
        if track[i+2] == -1 or track[i+3] == -1:
          continue
        offset1 = offsets[i]
        offset2 = offsets[i+1]
        idx1 = int(track[i+2]-offset1)
        idx2 = int(track[i+3]-offset2)
        pt1 = pts_mem[i][:2, idx1]
        pt2 = pts_mem[i+1][:2, idx2]
        p1 = (int(round(pt1[0])), int(round(pt1[1])))
        p2 = (int(round(pt2[0])), int(round(pt2[1])))
        cv2.line(out, p1, p2, clr, thickness=stroke, lineType=16)
        # Draw end points of each track.
        if i == N-2:
          clr2 = (255, 0, 0)
          cv2.circle(out, p2, stroke, clr2, -1, lineType=16)

class VideoStreamer(object):
  """ 
    Class to help process image streams. Three types of possible inputs:"
    1.) USB Webcam.
    2.) A directory of images (files in directory matching 'img_glob').
    3.) A video file, such as an .mp4 or .avi file.
  """
  def __init__(self, basedir, camid, height, width, skip, img_glob):
    self.cap = []
    self.camera = False
    self.video_file = False
    self.listing = []
    self.sizer = [height, width]
    self.i = 0
    self.skip = skip
    self.maxlen = 1000000
    # If the "basedir" string is the word camera, then use a webcam.
    if basedir == "camera/" or basedir == "camera":
      print('==> Processing Webcam Input.')
      self.cap = cv2.VideoCapture(camid)
      self.listing = range(0, self.maxlen)
      self.camera = True
    else:
      # Try to open as a video.
      self.cap = cv2.VideoCapture(basedir)
      lastbit = basedir[-4:len(basedir)]
      if (type(self.cap) == list or not self.cap.isOpened()) and (lastbit == '.mp4'):
        raise IOError('Cannot open movie file')
      elif type(self.cap) != list and self.cap.isOpened() and (lastbit != '.txt'):
        print('==> Processing Video Input.')
        num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.listing = range(0, num_frames)
        self.listing = self.listing[::self.skip]
        self.camera = True
        self.video_file = True
        self.maxlen = len(self.listing)
      else:
        print('==> Processing Image Directory Input.')
        search = os.path.join(basedir, img_glob)
        self.listing = glob.glob(search)
        self.listing.sort()
        self.listing = self.listing[::self.skip]
        self.maxlen = len(self.listing)
        if self.maxlen == 0:
          raise IOError('No images were found (maybe bad \'--img_glob\' parameter?)')

  def read_image(self, impath, img_size):
    """ Read image as grayscale and resize to img_size.
    Inputs
      impath: Path to input image.
      img_size: (W, H) tuple specifying resize size.
    Returns
      grayim: float32 numpy array sized H x W with values in range [0, 1].
    """
    grayim = cv2.imread(impath, 0)# read in grayscale mode.
    if grayim is None:
      raise Exception('Error reading image %s' % impath)
    # Image is resized via opencv.
    grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR) # 
    grayim = (grayim.astype('float32') / 255.)# This line scales these values to be in the range [0, 1] 
    return grayim

  def next_frame(self):
    """ Return the next frame, and increment internal counter.
    Returns
       image: Next H x W image.
       status: True or False depending whether image was loaded.
    """
    if self.i == self.maxlen:
      return (None, False)
    if self.camera:
      ret, input_image = self.cap.read()
      if ret is False:
        print('VideoStreamer: Cannot get image from camera (maybe bad --camid?)')
        return (None, False)
      if self.video_file:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])
      input_image = cv2.resize(input_image, (self.sizer[1], self.sizer[0]),
                               interpolation=cv2.INTER_AREA)
      input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
      input_image = input_image.astype('float')/255.0
    else:
      image_file = self.listing[self.i]
      input_image = self.read_image(image_file, self.sizer)
    # Increment internal counter.
    self.i = self.i + 1
    input_image = input_image.astype('float32')
    return (input_image, True)

# Function to compute robustness
def compute_robustness(pts1, pts2, homography_matrix, threshold):
    
    keypoints1 = []
    for i in range(pts1.shape[1]):
        x, y, _ = pts1[:, i]
        keypoint = cv2.KeyPoint(x, y, 1)  # Assuming size 1 for all keypoints
        keypoints1.append(keypoint)

    points1 = np.array([kp.pt for kp in keypoints1], dtype=np.float32)

    # print(f"number of keypoints1 inside compute_robustness function: {len(keypoints1)}")

    keypoints2 = []
    for i in range(pts2.shape[1]):
        x, y, _ = pts2[:, i]
        keypoint = cv2.KeyPoint(x, y, 1)  # Assuming size 1 for all keypoints
        keypoints2.append(keypoint)

    points2 = np.array([kp.pt for kp in keypoints2], dtype=np.float32)

    transformed_points1 = cv2.perspectiveTransform(np.expand_dims(points1, axis=1), homography_matrix)

    pts3 = []
    count = 0
    for i, transformed_point in enumerate(transformed_points1):
        min_dist = np.min(np.linalg.norm(points2 - transformed_point, axis=1))
        if min_dist < threshold:
            count += 1
            pts3.append(pts1[:, i])
    aa = np.array(pts3).T

    return aa, count / len(keypoints1)


if __name__ == '__main__':

  """
    Evaluation 1: Number of keypoints detected in single image
    Evaluation 2.1: Robustness of detector (rotation)
    Evaluation 2.2: Robustness of detector (scaling)
    Evaluation 2.3: Robustness of detector (blurred)
    Evaluation 3: Computational Efficiency
    Evaluation 4: Match ratio
    Evaluation 5: Match Score
    Evaluation 6: Distinctiveness
  """

  # Parse command line arguments.
  parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
  parser.add_argument('input', type=str, default='',#got it
      help='Image directory or movie file or "camera" (for webcam).')
  parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',#got it
      help='Path to pretrained weights file (default: superpoint_v1.pth).')
  parser.add_argument('--img_glob', type=str, default='*.png',#got it
      help='Glob match if directory of images is specified (default: \'*.png\').')
  parser.add_argument('--skip', type=int, default=1,#got it
      help='Images to skip if input is movie or directory (default: 1).')
  parser.add_argument('--show_extra', action='store_true',#got it
      help='Show extra debug outputs (default: False).')
  parser.add_argument('--H', type=int, default=120,#got it
      help='Input image height (default: 120).')
  parser.add_argument('--W', type=int, default=160,#got it
      help='Input image width (default:160).')
  parser.add_argument('--display_scale', type=int, default=1,#got it
      help='Factor to scale output visualization (default: 2).')
  parser.add_argument('--min_length', type=int, default=2,#got it
      help='Minimum length of point tracks (default: 2).')
  parser.add_argument('--max_length', type=int, default=5,#got it
      help='Maximum length of point tracks (default: 5).')
  parser.add_argument('--nms_dist', type=int, default=4,#got it
      help='Non Maximum Suppression (NMS) distance (default: 4).')
  parser.add_argument('--conf_thresh', type=float, default=0.015,#got it
      help='Detector confidence threshold (default: 0.015).')
  parser.add_argument('--nn_thresh', type=float, default=0.7,#got it
      help='Descriptor matching threshold (default: 0.7).')
  parser.add_argument('--camid', type=int, default=0,#got it
      help='OpenCV webcam video capture ID, usually 0 or 1 (default: 0).')
  parser.add_argument('--waitkey', type=int, default=1,#got it
      help='OpenCV waitkey time in ms (default: 1).')
  parser.add_argument('--cuda', action='store_true',#got it
      help='Use cuda GPU to speed up network processing speed (default: False)')
  parser.add_argument('--no_display', action='store_true',#got it
      help='Do not display images to screen. Useful if running remotely (default: False).')
  parser.add_argument('--write', action='store_true',#got it
      help='Save output frames to a directory (default: False)')
  parser.add_argument('--write_dir', type=str, default='tracker_outputs/',#got it
      help='Directory where to write output frames (default: tracker_outputs/).')
  opt = parser.parse_args()

  # This class helps load input images from different sources.
  vs = VideoStreamer(opt.input, opt.camid, opt.H, opt.W, opt.skip, opt.img_glob)

  print('==> Loading pre-trained network.')
  # This class runs the SuperPoint network and processes its outputs.
  fe = SuperPointFrontend(weights_path=opt.weights_path,
                          nms_dist=opt.nms_dist,
                          conf_thresh=opt.conf_thresh,
                          nn_thresh=opt.nn_thresh,
                          cuda=opt.cuda)
  
  print('==> Successfully loaded pre-trained network.')

  # This class helps merge consecutive point matches into tracks.
  tracker = PointTracker(opt.max_length, nn_thresh=fe.nn_thresh)

  # Create a window to display the demo.
  if not opt.no_display:
    win = 'SuperPoint Tracker'
    cv2.namedWindow(win)
  else:
    print('Skipping visualization, will not show a GUI.')

  # Font parameters for visualizaton.
  font = cv2.FONT_HERSHEY_DUPLEX
  font_clr = (255, 255, 255)
  font_pt = (4, 12)
  font_sc = 0.4

  # Create output directory if desired.
  if opt.write:
    print('==> Will write outputs to %s' % opt.write_dir)
    if not os.path.exists(opt.write_dir):
      os.makedirs(opt.write_dir)

  print('==> Running Demo.')

##############################################
  image_number = 0
  output_file = './Superpoint_cuda.csv'
  with open(output_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    # Write the header (column names) to the CSV file
    csv_writer.writerow(['image_number', 'evaluation_1','evaluation_2_1','evaluation_2_2', 
                         'evaluation_2_3','evaluation_3','evaluation_4','evaluation_5','evaluation_6'])
    
##############################################

# nubmer of csv lines = n-1( image_number<n )

    while image_number < 1145: 
      evaluation_1 = None
      evaluation_2_1 = None
      evaluation_2_2 = None
      evaluation_2_3 = None
      evaluation_3 = None
      evaluation_4 = None
      evaluation_5 = None
      evaluation_6 = None

      time1=None


      print('-----------------------------------------------------------------------')

      # Get a new image. the image is grayscale, resized to a specified size, and then scaled its pixel values to be in the range [0, 1].
      img, status = vs.next_frame()
      if status is False:
        break
      print("Processing image:", image_number)

      # Get points and descriptors.
      start1 = time.time()
      pts, desc, heatmap = fe.run(img)
      end1 = time.time()
      evaluation_1 = pts.shape[1]

      time1 = ((end1 - start1)*1000)

      ###################################################
      print("Evaluation 1: Number of keypoints detected in single image:", pts.shape[1])
      ###################################################

      # Get the image shape (height, width)
      (height, width) = img.shape[:2]

      # Calculate the center of the image
      center = (width / 2, height / 2)

      # Generate the rotation matrix
      # The arguments are center, angle of rotation, and scale
      rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)

      # Calculate the new size of the image to avoid clipping
      new_width = int((height * abs(np.sin(np.radians(45))) + 
                      width * abs(np.cos(np.radians(45)))))
      new_height = int((height * abs(np.cos(np.radians(45))) + 
                        width * abs(np.sin(np.radians(45)))))

      # Adjust the rotation matrix
      rotation_matrix[0, 2] += (new_width / 2) - center[0]
      rotation_matrix[1, 2] += (new_height / 2) - center[1]

      # Convert the 2x3 rotation matrix to a 3x3 homography matrix
      rotated_homography_matrix = np.vstack([rotation_matrix, [0, 0, 1]])

      # Perform the rotation using warpPerspective
      # The arguments are source image, homography matrix, and output image shape
      rotated_img = cv2.warpPerspective(img, rotated_homography_matrix, (new_width, new_height))

      pts_rotated_img, desc_rotated_img, heatmap_rotated_img = fe.run(rotated_img)

      # Set the threshold
      threshold_111 = 4.0

      # Compute the robustness
      c1,robustness_rotation = compute_robustness(pts, pts_rotated_img, rotated_homography_matrix, threshold_111)

      evaluation_2_1 = robustness_rotation
      print(f"Evaluation 2.1: Robustness of detector (rotation): {robustness_rotation*100} %")
      #########################################################

      scaling_factor = 2.0

      scaled_img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)

      pts_scaled_img, desc_scaled_img, heatmap_scaled_img = fe.run(scaled_img)

      # Calculate the homography matrix for scaling
      scaling_homography_matrix = np.array([
          [scaling_factor, 0, 0],
          [0, scaling_factor, 0],
          [0, 0, 1]
      ])

      # Compute the robustness
      c2,robustness_scaled = compute_robustness(pts, pts_scaled_img, scaling_homography_matrix, threshold_111)

      evaluation_2_2 = robustness_scaled
      print(f"Evaluation 2.2: Robustness of detector (scaling): {robustness_scaled*100} %")
      #######################################################

      # Blurring parameters
      blur_kernel_size = (5, 5)
      blur_sigma = 0

      # Apply Gaussian blur to the image
      blurred_img = cv2.GaussianBlur(img, blur_kernel_size, blur_sigma)

      pts_blurred_img, desc_blurred_img, heatmap_blurred_img = fe.run(blurred_img)

      # Calculate the homography matrix for scaling
      blurred_homography_matrix = np.array([
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]
      ])

      # Compute the robustness
      c3,robustness_blurred = compute_robustness(pts, pts_blurred_img, blurred_homography_matrix, threshold_111)

      evaluation_2_3 = robustness_blurred
      print(f"Evaluation 2.3: Robustness of detector (blurred): {robustness_blurred*100} %")
      #########################################################
      # Add points and descriptors to the tracker.

      time2, evaluation_4, evaluation_5, evaluation_6 = tracker.update(pts, desc)
      evaluation_3 = time1+time2

      print("Evaluation 3: Computational Efficiency:", evaluation_3)
      print("Evaluation 4: Match Ratio:", evaluation_4)
      print("Evaluation 5: Match Score:", evaluation_5)
      print("Evaluation 6: Distinctiveness: " , evaluation_6)


      if image_number != 0:
        csv_writer.writerow([image_number, evaluation_1, evaluation_2_1, evaluation_2_2, 
                             evaluation_2_3, evaluation_3, evaluation_4, evaluation_5, evaluation_6])

      # Get tracks for points which were match successfully across all frames.
      tracks = tracker.get_tracks(opt.min_length)

      # Primary output - Show point tracks overlayed on top of input image.
      out1 = (np.dstack((img, img, img)) * 255.).astype('uint8')
      tracks[:, 1] /= float(fe.nn_thresh) # Normalize track scores to [0,1].
      tracker.draw_tracks(out1, tracks)
      if opt.show_extra:
        cv2.putText(out1, 'Point Tracks', font_pt, font, font_sc, font_clr, lineType=16)

      # Extra output -- Show current point detections.
      out2 = (np.dstack((img, img, img)) * 255.).astype('uint8')
      for pt in pts.T:
        pt1 = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(out2, pt1, 1, (0, 255, 0), -1, lineType=16)
      cv2.putText(out2, 'Raw Point Detections', font_pt, font, font_sc, font_clr, lineType=16)

      # save one sample image
      if image_number == 3:
        filename = './superpoint_sample.png'
        cv2.imwrite(filename, out2)

      # Extra output -- Show the point confidence heatmap.
      if heatmap is not None:
        min_conf = 0.001
        heatmap[heatmap < min_conf] = min_conf
        heatmap = -np.log(heatmap)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + .00001)
        out3 = myjet[np.round(np.clip(heatmap*10, 0, 9)).astype('int'), :]
        out3 = (out3*255).astype('uint8')
      else:
        out3 = np.zeros_like(out2)
      cv2.putText(out3, 'Raw Point Confidences', font_pt, font, font_sc, font_clr, lineType=16)

      # Resize final output.
      if opt.show_extra:
        out = np.hstack((out1, out2, out3))
        out = cv2.resize(out, (3*opt.display_scale*opt.W, opt.display_scale*opt.H))
      else:
        out = cv2.resize(out1, (opt.display_scale*opt.W, opt.display_scale*opt.H))

      # Display visualization image to screen.
      if not opt.no_display:
        cv2.imshow(win, out)

        key = cv2.waitKey(opt.waitkey) & 0xFF
        if key == ord('q'):
          print('Quitting, \'q\' pressed.')
          break

      # Optionally write images to disk.
      if opt.write:
        out_file = os.path.join(opt.write_dir, 'frame_%05d.png' % vs.i)
        print('Writing image to %s' % out_file)
        cv2.imwrite(out_file, out)

      image_number =  image_number + 1

    # Close any remaining windows.
    cv2.destroyAllWindows()

    print('==> Finshed Demo.')