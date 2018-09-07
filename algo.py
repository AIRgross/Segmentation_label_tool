"""
GUI for selecting pixel coordinates in a directory of images
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import cv2
from skimage.segmentation import  slic
from skimage.segmentation import mark_boundaries
from PIL import Image






class Selector:
  """
  Select pixel coordinates in image(s) using a mouse
  """

  def __init__(self, img_dir):
    
    self.img_dir = img_dir
    self.excel_path = os.path.join(img_dir,'coords.xlsx')

    self.coords = []
    self.num = []
    self.labelled_files = []

    self.keep_going = True

  def find_images(self):
    """
    Looks for JPG images in the image directory, will exit if None
    """

    allowed_extenstions = ['jpg', 'png']
    file_names = [fn for fn in os.listdir(self.img_dir)
                  if any(fn.lower().endswith(ext) for ext in allowed_extenstions)]

    if len(file_names) == 0:
      print('no images found in: \n'+self.img_dir)
      sys.exit(1)

    file_paths = [os.path.join(self.img_dir,fn) for fn in file_names]
    self.unlabelled_files = file_paths

  def load_stats(self):
    """
    Loads from where you left off if doing manual labor in batches
    """

    if os.path.isfile(self.excel_path):
      
      df = pd.read_excel(self.excel_path)
      self.labelled_files = [os.path.join(self.img_dir,x) for x in list(np.array(df['fileID']))]
#      self.unlabelled_files = [x for x in self.unlabelled_files if x not in self.labelled_files]

      self.coords = [json.loads(c) for c in list(np.array(df['coords']))]
      self.num = list(np.array(df['num']))

    return

  def save_stats(self):
    """
    Continously save the data
    """
    
    fileID = os.path.basename(self.img_path)
    string_coords = json.dumps(self.these_coords)
    num = len(self.these_coords)
    
    self.labelled_files.append(fileID)
    self.coords.append(string_coords)
    self.num.append(num)
    
    df = pd.DataFrame({
      'fileID':self.labelled_files,
      'num':self.num,
      'coords':self.coords
      })
    
    df.to_excel(self.excel_path, columns=['fileID','num','coords'], index=None)


  def find_superpixels_under_marking(marking, segments_slic):
    fg_segments = np.unique(segments_slic[marking[:,:,0]==255])
    return fg_segments

# Get a bool mask of the pixels for a given selection of superpixel IDs
  def pixels_for_segment_selection(superpixels_labels, selection):
    pixels_mask = np.where(np.isin(superpixels_labels, selection), True, False)
    return pixels_mask

 


  def image_handler(self):
    """
    Opens an image, draws red dots, save coords
    """
    
    self.these_coords = []
    
    # mouse callback function
    def red_dot(event,x,y,flags,param):
      if event == cv2.EVENT_LBUTTONUP:
        self.these_coords.append((x,y)) 
        cv2.circle(img,(x,y),5,(255,255,255),-1)
        cv2.circle(img,(x,y),4,(0,0,255),-1)
        img[y,x,0]=-255
        print((x,y))
    # interactive display
    img_ref = cv2.imread(self.img_path)
    img_ref=np.asarray(img_ref)
    img=img_ref

    segments_slic = slic(img_ref, n_segments=50, compactness=20 , sigma=1)   
   #  segments_slic = felzenszwalb(img, scale=1000, sigma=5, min_size=50)
    
    img=mark_boundaries(img_ref, segments_slic)
    clone = img.copy()

    cv2.namedWindow('pixel selector', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('pixel selector',red_dot)

    segmask_ref=segments_slic
    segmask=segments_slic
    fg_segments=segments_slic
    test=1
    # event handler

    while(test==1):
      cv2.namedWindow('pixel selector',cv2.WINDOW_NORMAL)
      cv2.resizeWindow('pixel selector',2800,1400) 
      cv2.imshow('pixel selector',np.absolute(np.concatenate((np.absolute(img), img_ref), axis=1)))

      key = cv2.waitKey(1) & 0xFF
   
      fg_segments = segments_slic[ img[:,:,0] ==-255]  
      segmask = np.where(np.isin(segments_slic, fg_segments), True, False) 

      mask = np.zeros(img.shape[:2], dtype = "uint8")
      mask[segmask] = 255
      img_ref = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
 #     cv2.imshow('test',mask)

  

      if key == 27 or key == ord('q'):
         self.keep_going = False
         test=0
      # refresh dots
      if key == ord('r'):
         self.these_coords = []
         img = clone.copy()
  
 
    print(os.path.splitext(self.img_path)[0])
    cv2.imwrite( (os.path.splitext(self.img_path)[0])+"_seg.png", np.uint8(segmask * 255))
    cv2.destroyAllWindows()


    return

  def run(self):
    """
    Loop through the available images
    """
    
    self.find_images()
    self.load_stats()

    for img_path in self.unlabelled_files:
      
      self.img_path = img_path
      self.image_handler()
      self.save_stats()

#     if not self.keep_going:
#        sys.exit(1)
    
    print('All images done!')

    return

def main():
  
  args = sys.argv[1:]
  
  if len(args) != 1:
    print('usage: python pixel_selector.py path_to_image_directory')
    sys.exit(1)
  
  img_dir = args[0]
  selector = Selector(img_dir)
  selector.run()

if __name__ == '__main__':
  main()

