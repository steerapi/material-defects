/* eslint-disable prefer-const */
import { MutableRefObject, useEffect, useRef, useState } from 'react'
import './App.css'
import cv, { } from "@techstark/opencv-js";
import { BugAntIcon, ArrowsRightLeftIcon } from '@heroicons/react/20/solid';
import { BugAntIcon as BugAntIconOutline } from '@heroicons/react/24/outline';

function classNames(...classes) {
  return classes.filter(Boolean).join(' ')
}

function App() {
  const imageCompareMatchesRef = useRef<HTMLCanvasElement>(null);
  const imageAlignedRef = useRef<HTMLCanvasElement>(null);
  const imageDiffRef = useRef<HTMLCanvasElement>(null);
  const imageDiffNegativeRef = useRef<HTMLCanvasElement>(null);
  const imageMaskRef = useRef<HTMLCanvasElement>(null);
  const imageAbnormalRef = useRef<HTMLCanvasElement>(null);
  const imageAbnormalOverlayRef = useRef<HTMLCanvasElement>(null);
  const [numPixels, setNumPixels] = useState<number>(0);
  const [numBoxes, setNumBoxes] = useState<number>(0);
  const [bboxes, setBboxes] = useState<any>([]);
  const [bboxesNegative, setBboxesNegative] = useState<any>([]);

  const [cleanImg, setCleanImg] = useState<any>(null);
  const [defectImg, setDefectImg] = useState<any>(null);
  const [imgAbnormalBinary, setImgAbnormalBinary] = useState<any>(null);
  const [imgAbnormalRGB, setImgAbnormalRGB] = useState<any>(null);
  const [imgAbnormal, setImgAbnormal] = useState<any>(null);
  const [imgAbnormalBinaryNegative, setImgAbnormalBinaryNegative] = useState<any>(null);
  const [imgAbnormalRGBNegative, setImgAbnormalRGBNegative] = useState<any>(null);
  const [imgAbnormalNegative, setImgAbnormalNegative] = useState<any>(null);

  
  const [img1, setImg1] = useState<HTMLImageElement | null>(null);
  const [img2, setImg2] = useState<HTMLImageElement | null>(null);
  const [img1URL, setImg1URL] = useState<any>(null);
  const [img2URL, setImg2URL] = useState<any>(null);
  const [threshold, setThreshold] = useState<number>(0);
  const [isDebugging, setIsDebugging] = useState<boolean>(false);
  const [mode, setMode] = useState<string>("defective");
  const [method, setMethod] = useState<string>("ORB");
  const [resolution, setResolution] = useState<number>(512);
  const [autoRatio, setAutoRatio] = useState<number>(0.8);

  const input1Ref = useRef<HTMLInputElement>(null);
  const input2Ref = useRef<HTMLInputElement>(null);

  function thresholdDefectsNegative(image_abnormal, threshold = 0) {
    // if (imgAbnormalBinaryNegative && !imgAbnormalBinaryNegative.isDeleted()) {
    //   imgAbnormalBinaryNegative.delete();
    // }

    // binarize image_abnormal
    let image_abnormal_binary = new cv.Mat();
    cv.threshold(image_abnormal, image_abnormal_binary, threshold, 255, cv.THRESH_BINARY);
    cv.imshow(imageDiffNegativeRef.current, image_abnormal_binary);

    // find location of abnormality in image_abnormal_binary
    let contours: any = new cv.MatVector();
    let hierarchy = new cv.Mat();
    cv.findContours(image_abnormal_binary, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    let bboxes = [];
    // for each contours, find the bounding rectangle
    for (let i = 0; i < contours.size(); ++i) {
      let cnt = contours.get(i);
      let rect = cv.boundingRect(cnt);
      let x = rect.x;
      let y = rect.y;
      let w = rect.width;
      let h = rect.height;

      // save bounding box coordinates with padding of 10
      bboxes.push([x - 10, y - 10, x + w + 10, y + h + 10]);
    }
    // merge overlapping boxes in bboxes to one bigger box 
    mergeBboxes(bboxes);
    setBboxesNegative(bboxes);

    setImgAbnormalBinaryNegative(image_abnormal_binary);

    // cleanup
    contours.delete();
    hierarchy.delete();
  }

  function thresholdDefects(image_abnormal, threshold = 0) {
    // binarize image_abnormal
    let image_abnormal_binary = new cv.Mat();
    cv.threshold(image_abnormal, image_abnormal_binary, threshold, 255, cv.THRESH_BINARY);
    cv.imshow(imageDiffRef.current, image_abnormal_binary);

    // find location of abnormality in image_abnormal_binary
    let contours: any = new cv.MatVector();
    let hierarchy = new cv.Mat();
    cv.findContours(image_abnormal_binary, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    let bboxes = [];
    // for each contours, find the bounding rectangle
    for (let i = 0; i < contours.size(); ++i) {
      let cnt = contours.get(i);
      let rect = cv.boundingRect(cnt);
      let x = rect.x;
      let y = rect.y;
      let w = rect.width;
      let h = rect.height;

      // save bounding box coordinates with padding of 10
      bboxes.push([x - 10, y - 10, x + w + 10, y + h + 10]);
    }
    // merge overlapping boxes in bboxes to one bigger box 
    mergeBboxes(bboxes);
    setBboxes(bboxes);

    if (imgAbnormalBinary && !imgAbnormalBinary.isDeleted()) {
      imgAbnormalBinary.delete();
    }
    setImgAbnormalBinary(image_abnormal_binary);

    // cleanup
    contours.delete();
    hierarchy.delete();
  }

  function autoThreshold(image_abnormal, ratio = 0.8) {
    let minMax: any = (cv as any).minMaxLoc(image_abnormal);
    let threshold = Math.floor((minMax.maxVal - minMax.minVal) * ratio);
    setThreshold(threshold);
    return threshold;
  }

  function mergeBboxes(bboxes: any[]) {
    let merged = true;
    while (merged) {
      merged = false;
      for (let i = 0; i < bboxes.length; i++) {
        let bbox1 = bboxes[i];
        for (let j = i + 1; j < bboxes.length; j++) {
          let bbox2 = bboxes[j];
          // if bbox1 and bbox2 overlap
          if (bbox1[0] < bbox2[2] && bbox1[2] > bbox2[0] && bbox1[1] < bbox2[3] && bbox1[3] > bbox2[1]) {
            // merge bbox1 and bbox2
            bbox1[0] = Math.min(bbox1[0], bbox2[0]);
            bbox1[1] = Math.min(bbox1[1], bbox2[1]);
            bbox1[2] = Math.max(bbox1[2], bbox2[2]);
            bbox1[3] = Math.max(bbox1[3], bbox2[3]);
            // remove bbox2
            bboxes.splice(j, 1);
            j--;

            merged = true;
          }
        }
      }
    }
  }

  function computeDefects(threshold = 0, method = "ORB", resolution = "512", knnDistance_option = 0.7) {
    // set seed to keep result consistent
    cv.setRNGSeed(0);

    if (imgAbnormalBinary && !imgAbnormalBinary.isDeleted()) {
      imgAbnormalBinary.delete();
    }
    if (imgAbnormalRGB && !imgAbnormalRGB.isDeleted()) {
      imgAbnormalRGB.delete();
    }
    if (imgAbnormal && !imgAbnormal.isDeleted()) {
      imgAbnormal.delete();
    }
    if (cleanImg && !cleanImg.isDeleted()) {
      cleanImg.delete();
    }
    if (defectImg && !defectImg.isDeleted()) {
      defectImg.delete();
    }

    // load images
    let im1 = cv.imread(img1);
    // save original image size
    // let originalSize1 = new cv.Size(im1.cols, im1.rows);

    // resize to w=1024, keep same aspect ratio
    // calculate new height
    if (resolution !== "original") {
      let newW = parseInt(resolution)
      let newHeight = Math.round((newW / im1.cols) * im1.rows);
      let im1Resized = new cv.Mat();
      cv.resize(im1, im1Resized, new cv.Size(newW, newHeight), 0, 0, cv.INTER_AREA);

      // resize back to originalSize
      // let im1Resized2 = new cv.Mat();
      // cv.resize(im1Resized, im1Resized2, originalSize1, 0, 0, cv.INTER_AREA);
      // im1Resized.delete();
      // im1Resized = im1Resized2;

      im1.delete();
      im1 = im1Resized;
    }
    console.log("im1", im1.size());
    // blur im1 to remove noise
    let im1Blur = new cv.Mat();
    cv.GaussianBlur(im1, im1Blur, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
    im1.delete();
    im1 = im1Blur;

    let im2 = cv.imread(img2);
    // save original image size
    // let originalSize2 = new cv.Size(im2.cols, im2.rows);

    // resize to w=1024, keep same aspect ratio
    // calculate new height
    if (resolution !== "original") {
      let newW = parseInt(resolution)
      let newHeight = Math.round((newW / im2.cols) * im2.rows);
      let im2Resized = new cv.Mat();
      cv.resize(im2, im2Resized, new cv.Size(newW, newHeight), 0, 0, cv.INTER_AREA);

      // resize back to originalSize
      // let im2Resized2 = new cv.Mat();
      // cv.resize(im2Resized, im2Resized2, originalSize2, 0, 0, cv.INTER_AREA);
      // im2Resized.delete();
      // im2Resized = im2Resized2;

      im2.delete();
      im2 = im2Resized;
    }
    console.log("im2", im2.size());
    // blur im2 to remove noise
    let im2Blur = new cv.Mat();
    cv.GaussianBlur(im2, im2Blur, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
    im2.delete();
    im2 = im2Blur;

    // Convert images to grayscale
    let im1Gray = new cv.Mat();
    let im2Gray = new cv.Mat();
    cv.cvtColor(im1, im1Gray, cv.COLOR_BGRA2GRAY);
    cv.cvtColor(im2, im2Gray, cv.COLOR_BGRA2GRAY);

    let keypoints1: any = new cv.KeyPointVector();
    let keypoints2: any = new cv.KeyPointVector();
    let descriptors1 = new cv.Mat();
    let descriptors2 = new cv.Mat();

    let orb;
    if (method === "AKAZE") {
      orb = new (cv as any).AKAZE();
      // orb.setDescriptorSize(256);
      // orb.setThreshold(0.0003);
    } else if (method === "BRISK") {
      orb = new (cv as any).BRISK(30, 4, 1.0);
    } else {
      orb = new (cv as any).ORB(4096, 1.2, 8);
    }

    let det1 = new cv.Mat();
    let det2 = new cv.Mat();

    orb.detectAndCompute(im1Gray, det1, keypoints1, descriptors1);
    orb.detectAndCompute(im2Gray, det2, keypoints2, descriptors2);

    // console.log("Total of ", keypoints1.size(), " keypoints1 (img to align) and ", keypoints2.size(), " keypoints2 (reference)");
    // console.log("here are the first 5 keypoints for keypoints1:");
    // for (let i = 0; i < keypoints1.size(); i++) {
    //   console.log("keypoints1: [", i, "]", keypoints1.get(i).pt.x, keypoints1.get(i).pt.y);
    //   if (i === 5) { break; }
    // }

    // Match features.
    let good_matches: any = new cv.DMatchVector();
    // let bf: any = new cv.BFMatcher(); 
    let bf = new cv.BFMatcher(cv.NORM_HAMMING, false);
    let matches: any = new cv.DMatchVectorVector();
    bf.knnMatch(descriptors1, descriptors2, matches, 4);

    let counter = 0;
    for (let i = 0; i < matches.size(); ++i) {
      let match = matches.get(i);
      let dMatch1 = match.get(0);
      let dMatch2 = match.get(1);
      //console.log("[", i, "] ", "dMatch1: ", dMatch1, "dMatch2: ", dMatch2);
      // ratio test
      if (dMatch1.distance <= dMatch2.distance * knnDistance_option) {
        //console.log("***Good Match***", "dMatch1.distance: ", dMatch1.distance, "was less than or = to: ", "dMatch2.distance * parseFloat(knnDistance_option)", dMatch2.distance * parseFloat(knnDistance_option), "dMatch2.distance: ", dMatch2.distance, "knnDistance", knnDistance_option);
        good_matches.push_back(dMatch1);
        counter++;
      }
    }

    // console.log("keeping ", counter, " points in good_matches vector out of ", matches.size(), " contained in this match vector:", matches);
    // console.log("here are first 5 matches");
    // for (let t = 0; t < matches.size(); ++t) {
    //   console.log("[" + t + "]", "matches: ", matches.get(t));
    //   if (t === 5) { break; }
    // }

    // console.log("here are first 5 good_matches");
    // for (let r = 0; r < good_matches.size(); ++r) {
    //   console.log("[" + r + "]", "good_matches: ", good_matches.get(r));
    //   if (r === 5) { break; }
    // }

    // Draw top matches
    let imMatches = new cv.Mat();
    let color = new cv.Scalar(0, 255, 0, 255);
    cv.drawMatches(im1, keypoints1, im2, keypoints2,
      good_matches, imMatches, color);
    cv.imshow(imageCompareMatchesRef.current, imMatches);

    // Extract location of good matches
    let points1 = [];
    let points2 = [];
    for (let i = 0; i < good_matches.size(); i++) {
      points1.push(keypoints1.get(good_matches.get(i).queryIdx).pt.x);
      points1.push(keypoints1.get(good_matches.get(i).queryIdx).pt.y);
      points2.push(keypoints2.get(good_matches.get(i).trainIdx).pt.x);
      points2.push(keypoints2.get(good_matches.get(i).trainIdx).pt.y);
    }

    // original size
    let originalSize1 = new cv.Size(im1.cols, im1.rows);
    let originalSize2 = new cv.Size(im2.cols, im2.rows);

    // reload images
    im1.delete()
    im1 = cv.imread(img1);
    im2.delete()
    im2 = cv.imread(img2);

    // new size
    let newSize1 = new cv.Size(im1.cols, im1.rows);
    let newSize2 = new cv.Size(im2.cols, im2.rows);

    // adjust points coordinates to reflect resizing
    for (let i = 0; i < points1.length; i += 2) {
      points1[i] = points1[i] * newSize1.width / originalSize1.width;
      points1[i + 1] = points1[i + 1] * newSize1.height / originalSize1.height;
      points2[i] = points2[i] * newSize2.width / originalSize2.width;
      points2[i + 1] = points2[i + 1] * newSize2.height / originalSize2.height;
    }

    const mat1 = new cv.Mat(points1.length, 1, cv.CV_32FC2);
    mat1.data32F.set(points1);
    const mat2 = new cv.Mat(points2.length, 1, cv.CV_32FC2);
    mat2.data32F.set(points2);

    // Find homography
    let hMask = new cv.Mat();
    let h = cv.findHomography(mat1, mat2, cv.RHO, 5.0, hMask, 2000, 0.995);
    hMask.delete();

    if (h.empty()) {
      alert("homography matrix empty!");
    }
    else {
      console.log("h:", h);
      console.log("[", h.data64F[0], ",", h.data64F[1], ",", h.data64F[2]);
      console.log("", h.data64F[3], ",", h.data64F[4], ",", h.data64F[5]);
      console.log("", h.data64F[6], ",", h.data64F[7], ",", h.data64F[8], "]");
    }

    // Use homography to warp image
    let image_B_final_result = new cv.Mat();

    // // original size
    // let originalSize = new cv.Size(im2.cols, im2.rows);

    // // reload images
    // im1.delete()
    // im1 = cv.imread(img1);
    // im2.delete()
    // im2 = cv.imread(img2);

    // // new size
    // let newSize = new cv.Size(im2.cols, im2.rows);

    // // adjust homography matrix to account for the fact that the size has changed
    // h.data64F[0] = h.data64F[0] * (newSize.width / originalSize.width);
    // h.data64F[1] = h.data64F[1] * (newSize.width / originalSize.width);
    // h.data64F[2] = h.data64F[2] * (newSize.width / originalSize.width);
    // h.data64F[3] = h.data64F[3] * (newSize.height / originalSize.height);
    // h.data64F[4] = h.data64F[4] * (newSize.height / originalSize.height);
    // h.data64F[5] = h.data64F[5] * (newSize.height / originalSize.height);
    // h.data64F[6] = h.data64F[6] * (newSize.width / originalSize.width);
    // h.data64F[7] = h.data64F[7] * (newSize.height / originalSize.height);

    // wrap image with homography
    cv.warpPerspective(im1, image_B_final_result, h, im2.size());

    cv.imshow(imageAlignedRef.current, image_B_final_result);

    setCleanImg(image_B_final_result);
    setDefectImg(im2);

    // create a white image
    let white_image = cv.Mat.zeros(image_B_final_result.rows, image_B_final_result.cols, cv.CV_8UC3);
    white_image.setTo([255, 255, 255, 255]);
    // wrap white image with homography
    let white_image_warped = new cv.Mat();
    cv.warpPerspective(white_image, white_image_warped, h, im2.size());

    // convert white_image_warped to grayscale
    let white_image_warped_gray = new cv.Mat();
    cv.cvtColor(white_image_warped, white_image_warped_gray, cv.COLOR_RGBA2GRAY, 0);
    // binarize white_image_warped_gray
    cv.threshold(white_image_warped_gray, white_image_warped_gray, 0, 255, cv.THRESH_BINARY_INV);
    // expand the black pixels of white_image_warped_gray
    let kernel = cv.Mat.ones(15, 15, cv.CV_8U);
    cv.dilate(white_image_warped_gray, white_image_warped_gray, kernel);
    kernel.delete();

    // remove pixel where white_image_warped_gray is white
    // cv.bitwise_not(white_image_warped_gray, white_image_warped_gray);

    cv.imshow(imageMaskRef.current, white_image_warped_gray);

    // normalize image_B_final_result and im2
    // 1. intensity normalization
    let image_B_final_result_normalized = new cv.Mat();
    cv.normalize(image_B_final_result, image_B_final_result_normalized, 0, 255, cv.NORM_MINMAX, cv.CV_8UC3);
    let im2_normalized = new cv.Mat();
    cv.normalize(im2, im2_normalized, 0, 255, cv.NORM_MINMAX, cv.CV_8UC3);
    // 2. histrogram equalization
    // let image_B_final_result_normalized_hist = new cv.Mat();
    // cv.cvtColor(image_B_final_result_normalized, image_B_final_result_normalized_hist, cv.COLOR_BGR2HSV, 0);
    // let image_B_final_result_normalized_hist_channels = new cv.MatVector();
    // cv.split(image_B_final_result_normalized_hist, image_B_final_result_normalized_hist_channels);
    // cv.equalizeHist(image_B_final_result_normalized_hist_channels.get(2), image_B_final_result_normalized_hist_channels.get(2));
    // cv.merge(image_B_final_result_normalized_hist_channels, image_B_final_result_normalized_hist);
    // cv.cvtColor(image_B_final_result_normalized_hist, image_B_final_result_normalized_hist, cv.COLOR_HSV2BGR, 0);

    // let im2_normalized_hist = new cv.Mat();
    // cv.cvtColor(im2_normalized, im2_normalized_hist, cv.COLOR_BGR2HSV, 0);
    // let im2_normalized_hist_channels = new cv.MatVector();
    // cv.split(im2_normalized_hist, im2_normalized_hist_channels);
    // cv.equalizeHist(im2_normalized_hist_channels.get(2), im2_normalized_hist_channels.get(2));
    // cv.merge(im2_normalized_hist_channels, im2_normalized_hist);
    // cv.cvtColor(im2_normalized_hist, im2_normalized_hist, cv.COLOR_HSV2BGR, 0);

    // 3. gamma correction
    // let image_B_final_result_normalized_hist_gamma = new cv.Mat();
    // let im2_normalized_hist_gamma = new cv.Mat();
    // let gamma = 0.5;
    // let inverse_gamma = 1.0 / gamma;
    // let lut = new cv.Mat(1, 256, cv.CV_8UC1);
    // for (let i = 0; i < 256; i++) {
    //   lut.data[i] = Math.pow(i / 255.0, inverse_gamma) * 255.0;
    // }
    // cv.LUT(image_B_final_result_normalized_hist, lut, image_B_final_result_normalized_hist_gamma);
    // cv.LUT(im2_normalized_hist, lut, im2_normalized_hist_gamma);



    // find the abnormality in RGB channel of im2 and image_B_final_result
    // get rgb channels of im2
    // let im2_channels = new cv.MatVector();
    // cv.split(im2_normalized, im2_channels);

    // let image_B_final_result_channels = new cv.MatVector();
    // cv.split(image_B_final_result_normalized, image_B_final_result_channels);

    // convert im2_normalized, image_B_final_result_normalized to gray scale single channel
    let im2_gray = new cv.Mat();
    cv.cvtColor(im2_normalized, im2_gray, cv.COLOR_BGR2GRAY, 0);
    let image_B_final_result_gray = new cv.Mat();
    cv.cvtColor(image_B_final_result_normalized, image_B_final_result_gray, cv.COLOR_BGR2GRAY, 0);

    // blur
    cv.GaussianBlur(im2_gray, im2_gray, new cv.Size(31, 31), 0, 0, cv.BORDER_DEFAULT);
    cv.GaussianBlur(image_B_final_result_gray, image_B_final_result_gray, new cv.Size(31, 31), 0, 0, cv.BORDER_DEFAULT);

    let image_abnormal = new cv.Mat();
    let image_abnormal_neg = new cv.Mat();
    // cv.absdiff(im2_gray, image_B_final_result_gray, image_abnormal);
    // subtract the two images and split positive and negative values
    cv.subtract(image_B_final_result_gray, im2_gray, image_abnormal);
    cv.subtract(im2_gray, image_B_final_result_gray, image_abnormal_neg);
    
    image_B_final_result_gray.delete();
    im2_gray.delete();

    // blur
    cv.GaussianBlur(image_abnormal, image_abnormal, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);

    // reduce lines features
    const kernel2 = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
    cv.morphologyEx(image_abnormal, image_abnormal, cv.MORPH_OPEN, kernel2);
    kernel2.delete();

    // remove edges from image_abnormal
    let white_image_warped_gray_not = new cv.Mat();
    cv.bitwise_not(white_image_warped_gray, white_image_warped_gray_not);
    cv.bitwise_and(image_abnormal, white_image_warped_gray_not, image_abnormal);
    white_image_warped_gray_not.delete();

    // // replace edges of image_abnormal with mean of image_abnormal
    // let mean = cv.mean(image_abnormal);
    // let white_image_warped_gray_mean = new cv.Mat(white_image_warped_gray.rows, white_image_warped_gray.cols, white_image_warped_gray.type(), mean);
    // cv.add(image_abnormal, white_image_warped_gray_mean, image_abnormal);

    cv.imshow(imageAbnormalRef.current, image_abnormal);

    if (threshold === 0) {
      // find threshold
      // calculate max of image_abnormal
      threshold = autoThreshold(image_abnormal, autoRatio);
    }

    thresholdDefects(image_abnormal, threshold);
    
    // copy im2 to image_abnormal_binary_rgb
    let image_abnormal_binary_rgb = im2.clone();
    setImgAbnormalRGB(image_abnormal_binary_rgb);
    setImgAbnormal(image_abnormal);

    // clean up
    im1.delete();
    // im2.delete();

    im1Gray.delete();
    im2Gray.delete();
    keypoints1.delete();
    keypoints2.delete();
    descriptors1.delete();
    descriptors2.delete();
    orb.delete()
    det1.delete();
    det2.delete();
    good_matches.delete();
    matches.delete();
    imMatches.delete();

    mat1.delete();
    mat2.delete();
    h.delete();

    // image_B_final_result.delete();

    white_image.delete();
    white_image_warped.delete();
    white_image_warped_gray.delete();

    image_B_final_result_normalized.delete();
    im2_normalized.delete();

    // image_abnormal.delete();

  }

  const renderBboxesOverlayClean = () => {
    if (cleanImg.isDeleted()) return;

    // create a new copy of imgAbnormalRGB
    let imgCleanOverlay = cleanImg.clone();
    // draw bounding boxes
    for (let i = 0; i < bboxes.length; i++) {
      let bbox = bboxes[i];
      cv.rectangle(imgCleanOverlay, { x: bbox[0], y: bbox[1] }, { x: bbox[2], y: bbox[3] }, [255, 0, 0, 255], 2, cv.LINE_AA, 0);
    }

    cv.imshow(imageAbnormalOverlayRef.current, imgCleanOverlay);
    // imgAbnormalRGBOverlay.delete();
    imgCleanOverlay.delete();
  }

  const renderBboxesOverlay = () => {
    if (!imageAbnormalOverlayRef || !imageAbnormalOverlayRef.current || !imgAbnormalRGB || !imgAbnormalBinary) return;
    if (imgAbnormalRGB.isDeleted()) return;
    if (imgAbnormalBinary.isDeleted()) return;

    console.log("bboxes", bboxes.length)
    // create a new copy of imgAbnormalRGB
    let imgAbnormalRGBOverlay = imgAbnormalRGB.clone();
    // draw bounding boxes
    for (let i = 0; i < bboxes.length; i++) {
      let bbox = bboxes[i];
      cv.rectangle(imgAbnormalRGBOverlay, { x: bbox[0], y: bbox[1] }, { x: bbox[2], y: bbox[3] }, [255, 0, 0, 255], 2, cv.LINE_AA, 0);
    }

    cv.imshow(imageAbnormalOverlayRef.current, imgAbnormalRGBOverlay);
    // imgAbnormalRGBOverlay.delete();

    // clone imgAbnormalBinary
    let imgAbnormalBinaryAdjustment = imgAbnormalBinary.clone();

    // keep only the pixels in imgAbnormalBinaryAdjustment that are within the bounding boxes
    let bboxMask = cv.Mat.zeros(imgAbnormalBinaryAdjustment.rows, imgAbnormalBinaryAdjustment.cols, cv.CV_8U);
    for (let i = 0; i < bboxes.length; i++) {
      let bbox = bboxes[i];
      // create a mask of the bbox
      cv.rectangle(bboxMask, { x: bbox[0], y: bbox[1] }, { x: bbox[2], y: bbox[3] }, [255, 255, 255, 255], -1, cv.LINE_AA, 0);
    }
    // apply the mask to imgAbnormalBinaryAdjustment
    cv.bitwise_and(imgAbnormalBinaryAdjustment, bboxMask, imgAbnormalBinaryAdjustment);
    // imgAbnormalBinaryAdjustment.delete();

    // count number of pixels in image_abnormal_binary
    let num_pixels = cv.countNonZero(imgAbnormalBinaryAdjustment);
    setNumPixels(num_pixels);
    console.log("num_pixels", num_pixels);

    setNumBoxes(bboxes.length);
    console.log("num_boxes", bboxes.length);

    // clean up
    imgAbnormalRGBOverlay.delete();
    bboxMask.delete();
    imgAbnormalBinaryAdjustment.delete();
  }

  useEffect(() => {
    if (mode === "clean") {
      renderBboxesOverlayClean();
    } else {
      renderBboxesOverlay();
    }
  }, [bboxes, bboxesNegative])
  // useEffect(() => {
  //   if (!img1 || !img2) return;
  //   setThreshold(0);
  //   computeDefects(0);
  // }, [img1, img2])

  const canvasClickHandler = (e: any) => {
    console.log("canvas click", e);

    const rect = e.target.getBoundingClientRect()
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    // convert x to 0-1 value
    const x_norm = x / rect.width;
    // convert y to 0-1 value
    const y_norm = y / rect.height;

    console.log("x_norm", x_norm);
    console.log("y_norm", y_norm);

    // remove bboxes if clicked on
    for (let i = 0; i < bboxes.length; i++) {
      let bbox = bboxes[i];
      if (x_norm > bbox[0] / imgAbnormalBinary.cols && x_norm < bbox[2] / imgAbnormalBinary.cols && y_norm > bbox[1] / imgAbnormalBinary.rows && y_norm < bbox[3] / imgAbnormalBinary.rows) {
        console.log("remove bbox", bbox)
        bboxes.splice(i, 1);
        setBboxes([...bboxes]);
        break;
      }
    }

  }

  function downloadImage(imageAbnormalOverlayRef: MutableRefObject<HTMLCanvasElement>) {
    // download image from canvas
    const canvas = imageAbnormalOverlayRef.current;
    const image = canvas.toDataURL("image/png", 1.0).replace("image/png", "image/octet-stream");
    const link = document.createElement('a');
    // with time stamp in the name
    link.download = `image_${new Date().getTime()}.png`;
    link.href = image;
    link.click();
  }

  return (
    <>
      {/* sticky footer at the bottom */}
      <div className="flex flex-row fixed bottom-0 bg-gray-100 w-full px-4 justify-between z-10 space-x-2 items-center">
        {/* buttons to swap between the 3 canvases */}
        <div className="flex space-x-2 my-4 items-center">
          <button type="button"
            className={classNames(mode === 'clean' ? 'text-white bg-blue-500' : 'text-black bg-grey-500', "rounded-md border-0 px-3 py-2 text-sm font-semibold shadow-md hover:bg-gray-400")} onClick={() => {
              // show imageAbnormal
              setMode("clean");
              renderBboxesOverlayClean();
              // cv.imshow(imageAbnormalOverlayRef.current, cleanImg);
            }}>Clean</button>
          {/* <button type="button"
            className="rounded bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50" onClick={() => {
              // show imageDiff
              cv.imshow(imageAbnormalOverlayRef.current, defectImg);
            }}>Defective Image</button> */}
          <button type="button"
            className={classNames(mode === 'defective' ? 'text-white bg-blue-500' : 'text-black bg-grey-500', "rounded-md border-0 px-3 py-2 text-sm font-semibold shadow-md hover:bg-gray-400")} onClick={() => {
              // show imageAligned
              setMode("defective");
              renderBboxesOverlay();
            }}>Defective</button>
          {/* swap button to swap clean and defective image */}
          <button type="button"
            className="rounded bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50" onClick={() => {
              // swap url
              setImg1URL(img2URL);
              setImg2URL(img1URL);
              // swap clean and defective image
              setImg1(img2);
              setImg2(img1);

              // recompute
              computeDefects(0, method, resolution);
            }}>
            <ArrowsRightLeftIcon className="w-5 text-gray-500"
              aria-hidden="true"></ArrowsRightLeftIcon>
          </button>
        </div>
        {/* slider for threshold */}
        <div className="flex my-4">
          <div className="flex flex-col text-center">
            {/* button to minus threshold */}
            <div className="flex flex-row space-x-2 items-center">
              <button type="button"
                className="rounded-md border-0 text-white bg-blue-500 px-3 py-2 text-sm font-semibold shadow-md hover:bg-gray-400" onClick={() => {
                  setThreshold(threshold - 1);
                  thresholdDefects(imgAbnormal, threshold - 1);
                }}>-</button>
              {/* display threshold value */}
              <label htmlFor="threshold" className="text-gray-900 font-semibold">{threshold}</label>

              {/* button to plus threshold */}
              <button type="button"
                className="rounded-md border-0 text-white bg-blue-500 px-3 py-2 text-sm font-semibold shadow-md hover:bg-gray-400" onClick={() => {
                  setThreshold(threshold + 1);
                  thresholdDefects(imgAbnormal, threshold + 1);
                }}>+</button>
              {/* auto detect threshold button */}
              <button type="button"
                className="rounded-md border-0 text-white bg-blue-500 px-3 py-2 text-sm font-semibold shadow-md hover:bg-gray-400" onClick={() => {
                  setAutoRatio(0.6);
                  const t = autoThreshold(imgAbnormal, 0.6);
                  thresholdDefects(imgAbnormal, t);
                }}>0.6</button>
              <button type="button"
                className="rounded-md border-0 text-white bg-blue-500 px-3 py-2 text-sm font-semibold shadow-md hover:bg-gray-400" onClick={() => {
                  setAutoRatio(0.8);
                  const t = autoThreshold(imgAbnormal, 0.8);
                  thresholdDefects(imgAbnormal, t);
                }}>0.8</button>

            </div>
            <label htmlFor="threshold" className="text-gray-900 font-semibold">Threshold</label>
          </div>
        </div>
        {/* button to run the algorithm */}
        <div className="flex justify-center space-x-2 my-4">
          {/* select sizes: 256, 512, 1024, 2048, original */}
          <div className="flex flex-col items-center space-x-2">
            <select className="rounded-md border-0 text-white bg-blue-500 px-3 py-2 text-sm font-semibold shadow-md hover:bg-gray-400" onChange={(e) => {
              setResolution(e.target.value);
            }}
              value={resolution}>
              <option value="256">256</option>
              <option value="512">512</option>
              <option value="1024">1024</option>
              <option value="2048">2048</option>
              <option value="original">Original</option>
            </select>
            <label htmlFor="method" className="text-gray-900 font-semibold">Size</label>
          </div>
          <div className="flex flex-col items-center space-x-2">
            <select className="rounded-md border-0 text-white bg-blue-500 px-3 py-2 text-sm font-semibold shadow-md hover:bg-gray-400" onChange={(e) => {
              setMethod(e.target.value);
            }}
              value={method}>
              <option value="AKAZE">AKAZE (slow)</option>
              <option value="ORB">ORB (fast)</option>
            </select>
            <label htmlFor="method" className="text-gray-900 font-semibold">Method</label>
          </div>
          <button type="button"
            className="rounded-md border-0 text-white bg-blue-500 px-3 py-2 text-sm font-semibold shadow-md hover:bg-gray-400" onClick={() => {
              computeDefects(0, method, resolution);
            }}>Compute Defects</button>
        </div>
      </div>

      <div className="bg-gray-50">
        <header className="relative">
          <nav aria-label="Top">
            {/* Secondary navigation */}
            <div className="border-b border-gray-200 bg-white">
              <div className="mx-auto px-6">
                <div className="flex h-16 items-center justify-between">
                  <a>
                    <span>Compare two images for defects</span>
                  </a>
                  {/* bug icon to toggle setIsDebugging */}
                  <button type="button" className="inline-flex items-center p-2 border border-transparent rounded-full shadow-sm text-gray-400 hover:text-gray-500 bg-white hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500" onClick={() => {
                    setIsDebugging(!isDebugging);
                  }}>
                    {/* debug icon for debug */}
                    {isDebugging ?
                      <BugAntIcon
                        className="w-5 text-gray-500"
                        aria-hidden="true" />
                      :
                      <BugAntIconOutline
                        className="w-5 text-gray-500"
                        aria-hidden="true" />}
                  </button>
                </div>
              </div>
            </div>
          </nav>
        </header>

        <main className="mx-auto pb-24 pt-4 px-8">

          {/* Products */}



          {/* button to upload 2 images */}
          <div className="flex flex-row space-x-4">
            <div className="flex-1 flex flex-row space-x-4">
              <div>
                <label className="mb-1 block text-sm font-medium text-gray-700">Clean Image</label>
                {/* button to upload */}
                <button type="button"
                  className="rounded-md border-0 text-white bg-blue-500 px-3 py-2 text-sm font-semibold shadow-md hover:bg-gray-400" onClick={() => {
                    input1Ref.current?.click();
                  }}>Upload</button>
                {/* input file */}
                <input ref={input1Ref} type="file" className="hidden w-full text-sm file:mr-4 file:rounded-md file:border-0 file:bg-gray-500 file:py-2.5 file:px-4 file:text-sm file:font-semibold file:text-white hover:file:bg-primary-700 focus:outline-none disabled:pointer-events-none disabled:opacity-60" accept="image/*" onChange={(e: any) => {
                  // load img to objectURL
                  const img1 = e.target.files![0];
                  const img1URL = URL.createObjectURL(img1);
                  const imgElement = document.createElement('img');
                  imgElement.src = img1URL;
                  imgElement.onload = () => {
                    //   // const cvImg1 = cv.imread(imgElement);
                    //   // setCvImg1(cvImg1);
                    //   // cv.imshow(canvas1Ref.current, cvImg1);
                    //   // canvas2Ref.current.width = 300
                    //   // canvas2Ref.current.height = 300
                    setImg1(imgElement)
                  }
                  setImg1URL(img1URL);
                  // clear input value
                  e.target.value = null;
                }} />
              </div>
              <img className="h-24" src={img1URL}></img>
              {/* <canvas className="h-24"
                ref={canvas1Ref}></canvas> */}
            </div>
            <div className="flex-1 flex flex-row space-x-4">
              <div>
                <label className="mb-1 block text-sm font-medium text-gray-700">Defective Image</label>
                {/* button to upload */}
                <button type="button"
                  className="rounded-md border-0 text-white bg-blue-500 px-3 py-2 text-sm font-semibold shadow-md hover:bg-gray-400" onClick={() => {
                    input2Ref.current?.click();
                  }}>Upload</button>
                {/* input file */}
                <input ref={input2Ref} type="file" className="hidden w-full text-sm file:mr-4 file:rounded-md file:border-0 file:bg-gray-500 file:py-2.5 file:px-4 file:text-sm file:font-semibold file:text-white hover:file:bg-primary-700 focus:outline-none disabled:pointer-events-none disabled:opacity-60" accept="image/*" onChange={(e: any) => {
                  // load img to objectURL
                  const img2 = e.target.files![0];
                  const img2URL = URL.createObjectURL(img2);
                  const imgElement = document.createElement('img');
                  imgElement.src = img2URL;
                  imgElement.onload = () => {
                    //   // const cvImg2 = cv.imread(imgElement);
                    //   // setCvImg2(cvImg2);
                    //   // cv.imshow(canvas2Ref.current, cvImg2);
                    //   // canvas2Ref.current.width = 300
                    //   // canvas2Ref.current.height = 300
                    setImg2(imgElement);
                  }
                  setImg2URL(img2URL);
                  // clear input value
                  e.target.value = null;
                }} />
              </div>
              <img className="h-24" src={img2URL}></img>
              {/* <canvas className="h-24"
                ref={canvas2Ref}></canvas> */}
            </div>
          </div>

          <div className="flex flex-col space-y-1 mt-4 mb-2">
            {/* display numPixels */}
            <div className="text-sm font-semibold text-gray-900">Number of defect pixels: {numPixels}</div>
            {/* display numBoxes */}
            <div className="text-sm font-semibold text-gray-900">Number of defect bounding boxes: {numBoxes} <span className="font-normal">(click on the box to remove it)</span></div>
          </div>

          {/* canvas for display, the canvas should be on top of one another */}
          <div className="relative">
            {/* download image from canvas button */}
            <button type="button"
              className="absolute top-0 right-0 rounded-md border-0 text-white bg-blue-500 px-3 py-2 text-sm font-semibold shadow-md hover:bg-gray-400" onClick={() => {
                downloadImage(imageAbnormalOverlayRef);
              }
              }>Download</button>
            {/* canvas for displaying the image */}
            <canvas className="w-full max-h-screen-1/3"
              ref={imageAbnormalOverlayRef} onClick={canvasClickHandler}></canvas>
          </div>


          <div className={classNames(!isDebugging ? 'hidden' : '')}>
            <label>Matches</label>
            <canvas className="w-full" ref={imageCompareMatchesRef} id="imageCompareMatches" width="300" height="300"></canvas>
            <label>Aligned Image</label>
            <canvas className="w-full" ref={imageAlignedRef} id="imageAligned" width="300" height="300"></canvas>
            <label className="hidden">Aligned Mask</label>
            <canvas className="hidden w-full" ref={imageMaskRef} id="imageMaskRef" width="300" height="300"></canvas>
            <label>Absolute Difference</label>
            <canvas className="w-full" ref={imageAbnormalRef} id="imageAbnormal" width="300" height="300"></canvas>
            <label>Binary Threshold</label>
            <canvas className="w-full" ref={imageDiffRef} id="imageDiff" width="300" height="300"></canvas>
            <label>Binary Threshold Negative</label>
            <canvas className="w-full" ref={imageDiffNegativeRef} id="imageDiffNegative" width="300" height="300"></canvas>
          </div>
        </main>
      </div>
    </>
  )
}

export default App
