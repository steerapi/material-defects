/* eslint-disable prefer-const */
import { useEffect, useRef, useState } from 'react'
import './App.css'
import cv from "@techstark/opencv-js";
import { BugAntIcon, ArrowsRightLeftIcon, ArrowDownOnSquareIcon, PlayIcon } from '@heroicons/react/20/solid';
import { BugAntIcon as BugAntIconOutline } from '@heroicons/react/24/outline';
import { classNames } from './utils/react';
import { autoThreshold, downloadImage, histogramMatching, mergeBboxes, postProcessImageAbnormal } from './utils/vision';
import { useNavigate, useParams } from 'react-router-dom';
import { images, pairData, pairs } from './db/db';
import { loadImageElementFromURL } from './utils/file';
import { jsonToMat, matToJson } from './utils/cvmat';
import { placeholder } from './utils/image';

const autoOptions = [0, .25, .5, .8, .9, 1.0];

function App() {
  const { pairId } = useParams();
  const navigate = useNavigate();

  const imageCompareMatchesRef = useRef<HTMLCanvasElement>(null);
  const imageAlignedRef = useRef<HTMLCanvasElement>(null);
  const imageDiffRef = useRef<HTMLCanvasElement>(null);
  const imageDiffNegativeRef = useRef<HTMLCanvasElement>(null);
  const imageMaskRef = useRef<HTMLCanvasElement>(null);
  const imageAbnormalRef = useRef<HTMLCanvasElement>(null);
  const imageAbnormalNegRef = useRef<HTMLCanvasElement>(null);
  const imageAbnormalOverlayRef = useRef<HTMLCanvasElement>(null);

  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [numPixels, setNumPixels] = useState<number>(0);
  const [numBoxes, setNumBoxes] = useState<number>(0);
  const [numPixelsNegative, setNumPixelsNegative] = useState<number>(0);
  const [numBoxesNegative, setNumBoxesNegative] = useState<number>(0);
  const [bboxes, setBboxes] = useState<any>([]);
  const [bboxesNegative, setBboxesNegative] = useState<any>([]);

  const [cleanImg, setCleanImg] = useState<any>(null);
  const [defectImg, setDefectImg] = useState<any>(null);
  const [imgAbnormalBinary, setImgAbnormalBinary] = useState<any>(null);
  const [imgAbnormalRGB, setImgAbnormalRGB] = useState<any>(null);
  const [imgAbnormal, setImgAbnormal] = useState<any>(null);
  const [imgAbnormalBinaryNegative, setImgAbnormalBinaryNegative] = useState<any>(null);
  const [imgAbnormalNegative, setImgAbnormalNegative] = useState<any>(null);

  // const [img1, setImg1] = useState<HTMLImageElement | null>(null);
  // const [img2, setImg2] = useState<HTMLImageElement | null>(null);
  const [img1URL, setImg1URL] = useState<any>(null);
  const [img2URL, setImg2URL] = useState<any>(null);

  const [threshold, setThreshold] = useState<number>(0);
  const [thresholdNegative, setThresholdNegative] = useState<number>(0);

  const [mode, setMode] = useState<string>("defective");
  const [method, setMethod] = useState<string>("ORB");
  const [resolution, setResolution] = useState<string>("512");
  const [autoRatio, setAutoRatio] = useState<number>(0.8);
  const [autoRatioNegative, setAutoRatioNegative] = useState<number>(0.8);

  const [isDebugging, setIsDebugging] = useState<boolean>(false);
  const [isMergedBox, setIsMergedBox] = useState<boolean>(true);

  // useEffect(() => {
  //   (async () => {
  //     if (pairId) {
  //       await pairData.update(+pairId, {
  //         numPixels,
  //         numBoxes,
  //         numPixelsNegative,
  //         numBoxesNegative,
  //         threshold,
  //         thresholdNegative,
  //       });
  //     }
  //   })();
  // }, [numPixels, numBoxes, numPixelsNegative, numBoxesNegative, threshold, thresholdNegative]);
  // const input1Ref = useRef<HTMLInputElement>(null);
  // const input2Ref = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setTimeout(async () => {
      if (pairId) {
        const pair = await pairs.get(+pairId);
        const data = await pairData.get(+pairId);
        if (pair) {
          const imageData = await images.get(pair.cleanImageId);
          if (!imageData) {
            return;
          }
          const imageUrl = URL.createObjectURL(
            new Blob([imageData.buffer], { type: imageData.type })
          );
          setImg1URL(imageUrl);
          const imageData2 = await images.get(pair.defectiveImageId);
          if (!imageData2) {
            return;
          }
          const imageUrl2 = URL.createObjectURL(
            new Blob([imageData2.buffer], { type: imageData2.type })
          );
          setImg2URL(imageUrl2);

          // clear canvas
          if (imageAbnormalOverlayRef.current) {
            imageAbnormalOverlayRef.current.getContext('2d')?.clearRect(0, 0, imageAbnormalOverlayRef.current.width, imageAbnormalOverlayRef.current.height);
          }

          // reset state
          setNumPixels(0);
          setNumBoxes(0);
          setNumPixelsNegative(0);
          setNumBoxesNegative(0);
          setBboxes([]);
          setBboxesNegative([]);
          setCleanImg(null);
          setDefectImg(null);
          setImgAbnormalBinary(null);
          setImgAbnormalRGB(null);
          setImgAbnormal(null);
          setImgAbnormalBinaryNegative(null);
          setImgAbnormalNegative(null);
          setThreshold(0);
          setThresholdNegative(0);
          setMode("defective");
          setMethod("ORB");
          setResolution("512");
          setAutoRatio(0.8);
          setAutoRatioNegative(0.8);
          setIsDebugging(false);
          setIsMergedBox(true);

          console.log('pair found', pairId, data)
          if (data) {
            // load state
            try {
              await loadState();
            } catch (error) {
              console.error(error)
            }

          }

        } else {
          console.log('pair not found', pairId);
          navigate('/')
        }
      }
    })
  }, [pairId]);

  const saveState = async (isMergedBox, numPixels,
    numBoxes,
    numPixelsNegative,
    numBoxesNegative, cleanImg, defectImg, imgAbnormalRGB, imgAbnormalBinary, imgAbnormalBinaryNegative, bboxes, bboxesNegative, imgAbnormal, imgAbnormalNegative, threshold, thresholdNegative) => {
    const state = {
      isMergedBox,
      numPixels,
      numBoxes,
      numPixelsNegative,
      numBoxesNegative,
      bboxes: bboxes,
      bboxesNegative: bboxesNegative,
      cleanImg: matToJson(cleanImg),
      defectImg: matToJson(defectImg),
      // imgAbnormalRGB: matToJson(imgAbnormalRGB),
      // imgAbnormalBinary: matToJson(imgAbnormalBinary),
      // imgAbnormalBinaryNegative: matToJson(imgAbnormalBinaryNegative),
      imgAbnormal: matToJson(imgAbnormal),
      imgAbnormalNegative: matToJson(imgAbnormalNegative),
      threshold,
      thresholdNegative,
      mode,
      method,
      resolution,
      autoRatio,
      autoRatioNegative
    };
    // save to pairs db
    // await pairData.update(+pairId, { ...state });
    // update only state that is not null
    // await pairData.update(+pairId, Object.fromEntries(Object.entries(state).filter(([_, v]) => v != null)));
    // save to pairData db
    await pairData.put({ id: +pairId, ...Object.fromEntries(Object.entries(state).filter(([_, v]) => v != null)) });
  }

  const loadState = async () => {
    let data = await pairData.get(+pairId);
    if (data) {
      setNumPixels(data.numPixels || 0);
      setNumBoxes(data.numBoxes || 0);
      setNumPixelsNegative(data.numPixelsNegative || 0);
      setNumBoxesNegative(data.numBoxesNegative || 0);
      setIsMergedBox(data.isMergedBox ?? true);
      if (cleanImg && !cleanImg.isDeleted()) {
        cleanImg.delete();
      }
      setCleanImg(jsonToMat(data.cleanImg));
      if (defectImg && !defectImg.isDeleted()) {
        defectImg.delete();
      }
      setDefectImg(jsonToMat(data.defectImg));

      // if (imgAbnormalRGB && !imgAbnormalRGB.isDeleted()) {
      //   imgAbnormalRGB.delete();
      // }
      // setImgAbnormalRGB(jsonToMat(data.imgAbnormalRGB));
      // if (imgAbnormalBinary && !imgAbnormalBinary.isDeleted()) {
      //   imgAbnormalBinary.delete();
      // }
      // setImgAbnormalBinary(jsonToMat(data.imgAbnormalBinary));
      // if (imgAbnormalBinaryNegative && !imgAbnormalBinaryNegative.isDeleted()) {
      //   imgAbnormalBinaryNegative.delete();
      // }
      // setImgAbnormalBinaryNegative(jsonToMat(data.imgAbnormalBinaryNegative));

      if (imgAbnormal && !imgAbnormal.isDeleted()) {
        imgAbnormal.delete();
      }
      const img_abnormal = jsonToMat(data.imgAbnormal)
      setImgAbnormal(img_abnormal);
      if (imgAbnormalNegative && !imgAbnormalNegative.isDeleted()) {
        imgAbnormalNegative.delete();
      }
      const img_abnormal_neg = jsonToMat(data.imgAbnormalNegative)
      setImgAbnormalNegative(img_abnormal_neg);

      setThreshold(data.threshold);
      setThresholdNegative(data.thresholdNegative);

      setMode(data.mode || 'defective');
      setMethod(data.method || 'ORB');
      setResolution(data.resolution || 'original');
      setAutoRatio(data.autoRatio ?? 0.8);
      setAutoRatioNegative(data.autoRatioNegative ?? 0.8);

      if (img_abnormal) {
        thresholdDefects(img_abnormal, data.threshold);
      }
      if (img_abnormal_neg) {
        thresholdDefectsNegative(img_abnormal_neg, data.thresholdNegative);
      }

      setBboxes(data.bboxes || []);
      setBboxesNegative(data.bboxesNegative || []);

    }
  }

  function thresholdDefectsNegative(image_abnormal, threshold = 0) {
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
    if (isMergedBox) {
      mergeBboxes(bboxes);
    }
    setBboxesNegative(bboxes);
    pairData.update(+pairId, {
      bboxesNegative: bboxes
    });

    if (imgAbnormalBinaryNegative && !imgAbnormalBinaryNegative.isDeleted()) {
      imgAbnormalBinaryNegative.delete();
    }
    setImgAbnormalBinaryNegative(image_abnormal_binary);

    // cleanup
    contours.delete();
    hierarchy.delete();
    return [image_abnormal_binary, bboxes];
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
    if (isMergedBox) {
      mergeBboxes(bboxes);
    }
    setBboxes(bboxes);
    pairData.update(+pairId, {
      bboxes: bboxes
    });

    if (imgAbnormalBinary && !imgAbnormalBinary.isDeleted()) {
      imgAbnormalBinary.delete();
    }
    setImgAbnormalBinary(image_abnormal_binary);

    // cleanup
    contours.delete();
    hierarchy.delete();
    return [image_abnormal_binary, bboxes];
  }

  async function computeDefects(img1URL, img2URL, method = "ORB", resolution = "512", knnDistance_option = 0.7) {
    // check if already computed with the same parameters

    // set seed to keep result consistent
    cv.setRNGSeed(0);

    // if (imgAbnormalBinary && !imgAbnormalBinary.isDeleted()) {
    //   imgAbnormalBinary.delete();
    // }
    // if (imgAbnormal && !imgAbnormal.isDeleted()) {
    //   imgAbnormal.delete();
    // }
    // if (imgAbnormalBinaryNegative && !imgAbnormalBinaryNegative.isDeleted()) {
    //   imgAbnormalBinaryNegative.delete();
    // }
    // if (imgAbnormalNegative && !imgAbnormalNegative.isDeleted()) {
    //   imgAbnormalNegative.delete();
    // }
    // if (imgAbnormalRGB && !imgAbnormalRGB.isDeleted()) {
    //   imgAbnormalRGB.delete();
    // }
    // if (cleanImg && !cleanImg.isDeleted()) {
    //   cleanImg.delete();
    // }
    // if (defectImg && !defectImg.isDeleted()) {
    //   defectImg.delete();
    // }

    // load images
    let img1 = await loadImageElementFromURL(img1URL);
    let img2 = await loadImageElementFromURL(img2URL);

    let im1 = cv.imread(img1);
    if (resolution !== "original") {
      let newW = parseInt(resolution)
      let newHeight = Math.round((newW / im1.cols) * im1.rows);
      let im1Resized = new cv.Mat();
      cv.resize(im1, im1Resized, new cv.Size(newW, newHeight), 0, 0, cv.INTER_AREA);
      im1.delete();
      im1 = im1Resized;
    }
    // blur to remove noise
    let im1Blur = new cv.Mat();
    cv.GaussianBlur(im1, im1Blur, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
    im1.delete();
    im1 = im1Blur;
    // intensity normalization
    let im1Normalized = new cv.Mat();
    cv.normalize(im1, im1Normalized, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1);
    im1.delete();
    im1 = im1Normalized;

    let im2 = cv.imread(img2);
    if (resolution !== "original") {
      let newW = parseInt(resolution)
      let newHeight = Math.round((newW / im2.cols) * im2.rows);
      let im2Resized = new cv.Mat();
      cv.resize(im2, im2Resized, new cv.Size(newW, newHeight), 0, 0, cv.INTER_AREA);
      im2.delete();
      im2 = im2Resized;
    }
    // blur to remove noise
    let im2Blur = new cv.Mat();
    cv.GaussianBlur(im2, im2Blur, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
    im2.delete();
    im2 = im2Blur;
    // intensity normalization
    let im2Normalized = new cv.Mat();
    cv.normalize(im2, im2Normalized, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1);
    im2.delete();
    im2 = im2Normalized;

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

    // Match features.
    let good_matches: any = new cv.DMatchVector();
    // let bf: any = new cv.BFMatcher(); 
    let bf = new cv.BFMatcher(cv.NORM_HAMMING, false);
    let matches: any = new cv.DMatchVectorVector();
    // @ts-ignore
    bf.knnMatch(descriptors1, descriptors2, matches, 4);

    let counter = 0;
    for (let i = 0; i < matches.size(); ++i) {
      let match = matches.get(i);
      let dMatch1 = match.get(0);
      let dMatch2 = match.get(1);
      // ratio test
      if (dMatch1.distance <= dMatch2.distance * knnDistance_option) {
        good_matches.push_back(dMatch1);
        counter++;
      }
    }

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
    // im1.delete()
    // im1 = cv.imread(img1);
    // im2.delete()
    // im2 = cv.imread(img2);

    // new size
    let newSize1 = new cv.Size(im1.cols, im1.rows);
    let newSize2 = new cv.Size(im2.cols, im2.rows);

    // adjust points coordinates to reflect resizing
    // skip if original size is selected
    if (resolution !== "original") {
      for (let i = 0; i < points1.length; i += 2) {
        points1[i] = points1[i] * newSize1.width / originalSize1.width;
        points1[i + 1] = points1[i + 1] * newSize1.height / originalSize1.height;
        points2[i] = points2[i] * newSize2.width / originalSize2.width;
        points2[i + 1] = points2[i + 1] * newSize2.height / originalSize2.height;
      }
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
      alert("Homography not found. Cannot align images. Please try again.");
    }
    else {
      console.log("h:", h);
      console.log("[", h.data64F[0], ",", h.data64F[1], ",", h.data64F[2]);
      console.log("", h.data64F[3], ",", h.data64F[4], ",", h.data64F[5]);
      console.log("", h.data64F[6], ",", h.data64F[7], ",", h.data64F[8], "]");
    }

    // Use homography to warp image
    let image_B_final_result = new cv.Mat();

    // wrap image with homography
    cv.warpPerspective(im1, image_B_final_result, h, im2.size());
    cv.imshow(imageAlignedRef.current, image_B_final_result);

    if (cleanImg && !cleanImg.isDeleted()) {
      cleanImg.delete();
    }
    setCleanImg(image_B_final_result);
    if (defectImg && !defectImg.isDeleted()) {
      defectImg.delete();
    }
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

    // intensity normalization
    let image_B_final_result_normalized = new cv.Mat();
    cv.normalize(image_B_final_result, image_B_final_result_normalized, 0, 255, cv.NORM_MINMAX, cv.CV_8UC3);
    let im2_normalized = new cv.Mat();
    cv.normalize(im2, im2_normalized, 0, 255, cv.NORM_MINMAX, cv.CV_8UC3);

    // blur
    cv.GaussianBlur(image_B_final_result_normalized, image_B_final_result_normalized, { width: 5, height: 5 }, 0, 0, cv.BORDER_DEFAULT);
    cv.GaussianBlur(im2_normalized, im2_normalized, { width: 5, height: 5 }, 0, 0, cv.BORDER_DEFAULT);

    // histrogram normalization
    // let image_B_final_result_normalized_hist = new cv.Mat();
    // cv.cvtColor(image_B_final_result_normalized, image_B_final_result_normalized_hist, cv.COLOR_BGR2HSV, 0);
    // let image_B_final_result_normalized_hist_channels = new cv.MatVector();
    // cv.split(image_B_final_result_normalized_hist, image_B_final_result_normalized_hist_channels);
    // cv.equalizeHist(image_B_final_result_normalized_hist_channels.get(2), image_B_final_result_normalized_hist_channels.get(2));
    // cv.merge(image_B_final_result_normalized_hist_channels, image_B_final_result_normalized_hist);
    // cv.cvtColor(image_B_final_result_normalized_hist, image_B_final_result_normalized_hist, cv.COLOR_HSV2BGR, 0);
    // image_B_final_result_normalized.delete();
    // image_B_final_result_normalized = image_B_final_result_normalized_hist

    // let im2_normalized_hist = new cv.Mat();
    // cv.cvtColor(im2_normalized, im2_normalized_hist, cv.COLOR_BGR2HSV, 0);
    // let im2_normalized_hist_channels = new cv.MatVector();
    // cv.split(im2_normalized_hist, im2_normalized_hist_channels);
    // cv.equalizeHist(im2_normalized_hist_channels.get(2), im2_normalized_hist_channels.get(2));
    // cv.merge(im2_normalized_hist_channels, im2_normalized_hist);
    // cv.cvtColor(im2_normalized_hist, im2_normalized_hist, cv.COLOR_HSV2BGR, 0);
    // im2_normalized.delete();
    // im2_normalized = im2_normalized_hist;

    // histrogram matching image_B_final_result_normalized to im2_normalized    
    histogramMatching(image_B_final_result_normalized, im2_normalized, image_B_final_result_normalized);

    // convert im2_normalized, image_B_final_result_normalized to gray scale single channel
    let im2_gray = new cv.Mat();
    cv.cvtColor(im2_normalized, im2_gray, cv.COLOR_BGR2GRAY, 0);
    let image_B_final_result_gray = new cv.Mat();
    cv.cvtColor(image_B_final_result_normalized, image_B_final_result_gray, cv.COLOR_BGR2GRAY, 0);

    // blur
    cv.GaussianBlur(im2_gray, im2_gray, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
    cv.GaussianBlur(image_B_final_result_gray, image_B_final_result_gray, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);

    // median blur
    cv.medianBlur(im2_gray, im2_gray, 5);
    cv.medianBlur(image_B_final_result_gray, image_B_final_result_gray, 5);

    let image_abnormal = new cv.Mat();
    let image_abnormal_neg = new cv.Mat();
    // cv.absdiff(im2_gray, image_B_final_result_gray, image_abnormal);
    // subtract the two images and split positive and negative values
    cv.subtract(image_B_final_result_gray, im2_gray, image_abnormal);
    cv.subtract(im2_gray, image_B_final_result_gray, image_abnormal_neg);

    image_B_final_result_gray.delete();
    im2_gray.delete();

    // post process image_abnormal
    postProcessImageAbnormal(image_abnormal, white_image_warped_gray);
    postProcessImageAbnormal(image_abnormal_neg, white_image_warped_gray);

    // // replace edges of image_abnormal with mean of image_abnormal
    // let mean = cv.mean(image_abnormal);
    // let white_image_warped_gray_mean = new cv.Mat(white_image_warped_gray.rows, white_image_warped_gray.cols, white_image_warped_gray.type(), mean);
    // cv.add(image_abnormal, white_image_warped_gray_mean, image_abnormal);

    cv.imshow(imageAbnormalRef.current, image_abnormal);
    cv.imshow(imageAbnormalNegRef.current, image_abnormal_neg);

    // if (threshold === 0) {
    //   // find threshold
    //   // calculate max of image_abnormal
    //   threshold = autoThreshold(image_abnormal, autoRatio);
    // }

    let threshold = autoThreshold(image_abnormal, autoRatio);
    setThreshold(threshold);
    const [image_abnormal_binary, bboxes] = thresholdDefects(image_abnormal, threshold);
    if (imgAbnormal && !imgAbnormal.isDeleted()) {
      imgAbnormal.delete();
    }
    setImgAbnormal(image_abnormal);

    let thresholdNegative = autoThreshold(image_abnormal_neg, autoRatioNegative);
    setThresholdNegative(thresholdNegative);
    const [image_abnormal_binary_neg, bboxes_neg] = thresholdDefectsNegative(image_abnormal_neg, thresholdNegative);
    if (imgAbnormalNegative && !imgAbnormalNegative.isDeleted()) {
      imgAbnormalNegative.delete();
    }
    setImgAbnormalNegative(image_abnormal_neg);

    // copy im2 to image_abnormal_binary_rgb
    let image_abnormal_binary_rgb = im2.clone();
    if (imgAbnormalRGB && !imgAbnormalRGB.isDeleted()) {
      imgAbnormalRGB.delete();
    }
    setImgAbnormalRGB(image_abnormal_binary_rgb);

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

    const [numPixels, numBoxes] = calculateNumBoxes(image_abnormal_binary, bboxes);
    const [numPixelsNegative, numBoxesNegative] = calculateNumBoxesNegative(image_abnormal_binary_neg, bboxes_neg);

    // image_abnormal.delete();
    saveState(isMergedBox, numPixels,
      numBoxes,
      numPixelsNegative,
      numBoxesNegative, image_B_final_result, im2, image_abnormal_binary_rgb, image_abnormal_binary, image_abnormal_binary_neg, bboxes, bboxes_neg, image_abnormal, image_abnormal_neg, threshold, thresholdNegative);
  }

  const renderBboxesOverlayClean = () => {
    if (!cleanImg || cleanImg.isDeleted()) return;
    if (!imageAbnormalOverlayRef.current || !imageAbnormalOverlayRef.current) return;

    let imgCleanOverlay = cleanImg.clone();
    // draw bounding boxes
    for (let i = 0; i < bboxes.length; i++) {
      let bbox = bboxes[i];
      cv.rectangle(imgCleanOverlay, { x: bbox[0], y: bbox[1] }, { x: bbox[2], y: bbox[3] }, [255, 0, 0, 255], 2, cv.LINE_AA, 0);
    }
    // draw bboxesNegative
    for (let i = 0; i < bboxesNegative.length; i++) {
      let bbox = bboxesNegative[i];
      cv.rectangle(imgCleanOverlay, { x: bbox[0], y: bbox[1] }, { x: bbox[2], y: bbox[3] }, [0, 0, 255, 255], 2, cv.LINE_AA, 0);
    }

    cv.imshow(imageAbnormalOverlayRef.current, imgCleanOverlay);
    imgCleanOverlay.delete();
  }

  const renderBboxesOverlayDefect = () => {
    if (!defectImg || defectImg.isDeleted()) return;
    if (!imageAbnormalOverlayRef.current || !imageAbnormalOverlayRef.current) return;

    let imgDefectOverlay = defectImg.clone();
    // draw bounding boxes
    for (let i = 0; i < bboxes.length; i++) {
      let bbox = bboxes[i];
      cv.rectangle(imgDefectOverlay, { x: bbox[0], y: bbox[1] }, { x: bbox[2], y: bbox[3] }, [255, 0, 0, 255], 2, cv.LINE_AA, 0);
    }
    // draw bboxesNegative
    for (let i = 0; i < bboxesNegative.length; i++) {
      let bbox = bboxesNegative[i];
      cv.rectangle(imgDefectOverlay, { x: bbox[0], y: bbox[1] }, { x: bbox[2], y: bbox[3] }, [0, 0, 255, 255], 2, cv.LINE_AA, 0);
    }

    cv.imshow(imageAbnormalOverlayRef.current, imgDefectOverlay);
    imgDefectOverlay.delete();
  }

  const calculateNumBoxes = (imgAbnormalBinary, bboxes) => {
    if (!imgAbnormalBinary || imgAbnormalBinary.isDeleted()) return;
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
    // count number of pixels in image_abnormal_binary
    let num_pixels = cv.countNonZero(imgAbnormalBinaryAdjustment);
    setNumPixels(num_pixels);
    setNumBoxes(bboxes.length);
    pairData.update(+pairId, {
      numPixels: num_pixels,
      numBoxes: bboxes.length
    })
    imgAbnormalBinaryAdjustment.delete();
    bboxMask.delete();
    return [num_pixels, bboxes.length]
  }

  const calculateNumBoxesNegative = (imgAbnormalBinaryNegative, bboxesNegative) => {
    if (!imgAbnormalBinaryNegative || imgAbnormalBinaryNegative.isDeleted()) return;
    // for negatives
    let imgAbnormalBinaryAdjustmentNegative = imgAbnormalBinaryNegative.clone();
    // keep only the pixels in imgAbnormalBinaryAdjustment that are within the bounding boxes
    let bboxMaskNegative = cv.Mat.zeros(imgAbnormalBinaryAdjustmentNegative.rows, imgAbnormalBinaryAdjustmentNegative.cols, cv.CV_8U);
    for (let i = 0; i < bboxesNegative.length; i++) {
      let bbox = bboxesNegative[i];
      // create a mask of the bbox
      cv.rectangle(bboxMaskNegative, { x: bbox[0], y: bbox[1] }, { x: bbox[2], y: bbox[3] }, [255, 255, 255, 255], -1, cv.LINE_AA, 0);
    }
    // apply the mask to imgAbnormalBinaryAdjustment
    cv.bitwise_and(imgAbnormalBinaryAdjustmentNegative, bboxMaskNegative, imgAbnormalBinaryAdjustmentNegative);
    // count number of pixels in image_abnormal_binary
    let num_pixels_negative = cv.countNonZero(imgAbnormalBinaryAdjustmentNegative);
    setNumPixelsNegative(num_pixels_negative);
    setNumBoxesNegative(bboxesNegative.length);
    pairData.update(+pairId, {
      numPixelsNegative: num_pixels_negative,
      numBoxesNegative: bboxesNegative.length
    })
    imgAbnormalBinaryAdjustmentNegative.delete();
    bboxMaskNegative.delete();
    return [num_pixels_negative, bboxesNegative.length]
  }

  const canvasClickHandler = (e: any) => {
    const rect = e.target.getBoundingClientRect()
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    // convert x to 0-1 value
    const x_norm = x / rect.width;
    // convert y to 0-1 value
    const y_norm = y / rect.height;

    // remove bboxes if clicked on
    for (let i = 0; i < bboxes.length; i++) {
      let bbox = bboxes[i];
      if (x_norm > bbox[0] / imgAbnormalBinary.cols && x_norm < bbox[2] / imgAbnormalBinary.cols && y_norm > bbox[1] / imgAbnormalBinary.rows && y_norm < bbox[3] / imgAbnormalBinary.rows) {
        bboxes.splice(i, 1);
        setBboxes([...bboxes]);
        break;
      }
    }
    // remove bboxes if clicked on (negative)
    for (let i = 0; i < bboxesNegative.length; i++) {
      let bbox = bboxesNegative[i];
      if (x_norm > bbox[0] / imgAbnormalBinary.cols && x_norm < bbox[2] / imgAbnormalBinary.cols && y_norm > bbox[1] / imgAbnormalBinary.rows && y_norm < bbox[3] / imgAbnormalBinary.rows) {
        bboxesNegative.splice(i, 1);
        setBboxesNegative([...bboxesNegative]);
        break;
      }
    }

    // save
    pairData.update(+pairId, {
      bboxes: bboxes,
      bboxesNegative: bboxesNegative
    })

  }

  useEffect(() => {
    if (mode === "clean") {
      renderBboxesOverlayClean();
    } else {
      renderBboxesOverlayDefect();
    }
    if (!imgAbnormal) return;
    if (!imgAbnormalNegative) return;
    calculateNumBoxes(imgAbnormalBinary, bboxes);
    calculateNumBoxesNegative(imgAbnormalBinaryNegative, bboxesNegative);
  }, [bboxes, bboxesNegative])

  useEffect(() => {
    if (!imgAbnormal) return;
    if (!imgAbnormalNegative) return;
    thresholdDefects(imgAbnormal, threshold);
    thresholdDefectsNegative(imgAbnormalNegative, thresholdNegative);
  }, [isMergedBox]);

  return (
    <>
      {/* sticky footer at the bottom */}
      <div className="shadow-inner flex flex-row fixed bottom-0 bg-gray-100 w-full px-4 justify-between z-10 space-x-2 items-center overflow-x-scroll">

        {/* slider for threshold */}
        <div className="flex my-4">
          <div className="flex flex-col text-center space-y-2">
            {/* button to minus threshold */}
            <div className="flex flex-row space-x-2 items-center">
              <button type="button"
                className={classNames(mode === 'defective' ? 'text-white bg-red-500' : 'text-black bg-grey-500', "rounded-md border-0 px-3 py-2 text-xs font-semibold shadow-md hover:bg-gray-400 w-24")} onClick={() => {
                  // show imageAligned
                  setMode("defective");
                  renderBboxesOverlayDefect();
                  pairData.update(+pairId, {
                    mode: "defective"
                  });
                }}>Defective</button>
              <button type="button"
                className="rounded-md border-0 text-white bg-red-500 px-3 py-2 text-xs font-semibold shadow-md hover:bg-gray-400" onClick={() => {
                  setThreshold(threshold - 1);
                  thresholdDefects(imgAbnormal, threshold - 1);
                  // update pairs threshold
                  pairData.update(+pairId, {
                    threshold: threshold - 1
                  });
                }}>-</button>
              {/* display threshold value */}
              <label htmlFor="threshold" className="text-gray-900 font-semibold flex-grow">{threshold}</label>
              {/* button to plus threshold */}
              <button type="button"
                className="rounded-md border-0 text-white bg-red-500 px-3 py-2 text-xs font-semibold shadow-md hover:bg-gray-400" onClick={() => {
                  setThreshold(threshold + 1);
                  thresholdDefects(imgAbnormal, threshold + 1);
                  pairData.update(+pairId, {
                    threshold: threshold + 1
                  });
                }}>+</button>
              {/* auto detect threshold button */}
              <span className="isolate inline-flex rounded-md shadow-sm">
                {autoOptions.map((value, i) => (
                  <button
                    key={value}
                    type="button"
                    className={classNames(i == 0 && "relative inline-flex items-center rounded-l-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 ring-1 ring-inset ring-gray-300 hover:bg-gray-50 focus:z-10",
                      i > 0 && i < autoOptions.length - 1 && "relative -ml-px inline-flex items-center bg-white px-3 py-2 text-xs font-semibold text-gray-900 ring-1 ring-inset ring-gray-300 hover:bg-gray-50 focus:z-10",
                      i == autoOptions.length - 1 && "relative -ml-px inline-flex items-center rounded-r-md bg-white px-3 py-2 text-xs font-semibold text-gray-900 ring-1 ring-inset ring-gray-300 hover:bg-gray-50 focus:z-10"
                    )}
                    onClick={() => {
                      setAutoRatio(value);
                      const t = autoThreshold(imgAbnormal, value);
                      setThreshold(t);
                      thresholdDefects(imgAbnormal, t);
                      pairData.update(+pairId, {
                        threshold: t
                      });
                    }}
                  >
                    {value}
                  </button>
                ))}
              </span>

            </div>
            <div className="flex flex-row space-x-2 items-center">
              <button type="button"
                className={classNames(mode === 'clean' ? 'text-white bg-blue-500' : 'text-black bg-grey-500', "rounded-md border-0 px-3 py-2 text-xs font-semibold shadow-md hover:bg-gray-400 w-24")} onClick={() => {
                  // show imageAbnormal
                  setMode("clean");
                  renderBboxesOverlayClean();
                  pairData.update(+pairId, {
                    mode: "clean"
                  });
                  // cv.imshow(imageAbnormalOverlayRef.current, cleanImg);
                }}>Clean</button>
              <button type="button"
                className="rounded-md border-0 text-white bg-blue-500 px-3 py-2 text-xs font-semibold shadow-md hover:bg-gray-400" onClick={() => {
                  setThresholdNegative(thresholdNegative - 1);
                  thresholdDefectsNegative(imgAbnormalNegative, thresholdNegative - 1);
                  pairData.update(+pairId, {
                    thresholdNegative: thresholdNegative - 1
                  });
                }}>-</button>
              {/* display threshold value */}
              <label htmlFor="thresholdNegative" className="text-gray-900 font-semibold flex-grow">{thresholdNegative}</label>
              {/* button to plus threshold */}
              <button type="button"
                className="rounded-md border-0 text-white bg-blue-500 px-3 py-2 text-xs font-semibold shadow-md hover:bg-gray-400" onClick={() => {
                  setThresholdNegative(thresholdNegative + 1);
                  thresholdDefectsNegative(imgAbnormalNegative, thresholdNegative + 1);
                  pairData.update(+pairId, {
                    thresholdNegative: thresholdNegative + 1
                  });
                }}>+</button>
              {/* auto detect threshold button */}
              <span className="isolate inline-flex rounded-md shadow-sm">
                {autoOptions.map((value, i) => (
                  <button
                    key={value}
                    type="button"
                    className={classNames(i == 0 && "relative inline-flex items-center rounded-l-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 ring-1 ring-inset ring-gray-300 hover:bg-gray-50 focus:z-10",
                      i > 0 && i < autoOptions.length - 1 && "relative -ml-px inline-flex items-center bg-white px-3 py-2 text-xs font-semibold text-gray-900 ring-1 ring-inset ring-gray-300 hover:bg-gray-50 focus:z-10",
                      i == autoOptions.length - 1 && "relative -ml-px inline-flex items-center rounded-r-md bg-white px-3 py-2 text-xs font-semibold text-gray-900 ring-1 ring-inset ring-gray-300 hover:bg-gray-50 focus:z-10"
                    )}
                    onClick={() => {
                      setAutoRatioNegative(value);
                      const tN = autoThreshold(imgAbnormalNegative, value);
                      setThresholdNegative(tN);
                      thresholdDefectsNegative(imgAbnormalNegative, tN);
                      pairData.update(+pairId, {
                        thresholdNegative: tN
                      });
                    }}
                  >
                    {value}
                  </button>
                ))}
              </span>
            </div>
            {/* swap button to swap clean and defective image */}
            <div className='flex flex-row'>
              <button type="button"
                className="flex flex-row justify-center rounded bg-white px-3 py-2 text-xs font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50 w-24" onClick={() => {
                  // swap url
                  setImg1URL(img2URL);
                  setImg2URL(img1URL);
                  // swap clean and defective image
                  // setImg1(img2);
                  // setImg2(img1);
                  // reset bboxes
                  setBboxes([]);
                  setBboxesNegative([]);

                  // recompute

                  computeDefects(img2URL, img1URL, method, resolution);
                }}>
                <ArrowsRightLeftIcon className="w-3 text-gray-500"
                  aria-hidden="true"></ArrowsRightLeftIcon>
              </button>
              <label className="flex m-auto text-center text-gray-900 font-semibold">Threshold</label>
            </div>
          </div>
        </div>
        {/* button to run the algorithm */}
        <div className="flex justify-center space-x-2 my-4">
          {/* select sizes: 256, 512, 1024, 2048, original */}
          {/* <div className="flex flex-col items-center space-x-2">
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
          </div> */}
          <div className="flex flex-col items-center space-y-2">
            {/* disable merge bboxes checkbox */}
            <div className="flex flex-row items-center space-x-2">
              <input type="checkbox" className="rounded-md border-0 text-white bg-blue-500 px-3 py-2 text-sm font-semibold shadow-md hover:bg-gray-400" onChange={(e) => {
                setIsMergedBox(e.target.checked);
                pairData.update(+pairId, {
                  isMergedBox: e.target.checked
                });
              }}
                checked={isMergedBox}></input>
              <label htmlFor="method" className="text-gray-900 font-semibold">Merge Boxes</label>
            </div>
            <select className="w-32 rounded-md border-0 text-white bg-slate-500 px-3 py-2 text-xs font-semibold shadow-md hover:bg-gray-400" onChange={(e) => {
              setResolution(e.target.value);
            }}
              value={resolution}>
              <option value="256">256</option>
              <option value="512">512</option>
              <option value="1024">1024</option>
              <option value="2048">2048</option>
              <option value="original">Original</option>
            </select>
            <select className="w-32 rounded-md border-0 text-white bg-slate-500 px-3 py-2 text-xs font-semibold shadow-md hover:bg-gray-400" onChange={(e) => {
              setMethod(e.target.value);
            }}
              value={method}>
              <option value="AKAZE">AKAZE (slow)</option>
              <option value="ORB">ORB (fast)</option>
            </select>
            <div className='flex flex-row items-center space-x-2'>
              <label htmlFor="method" className="text-gray-900 font-semibold">Method</label>
              <button
                type="button"
                className="bg-white rounded-br px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50" onClick={() => {
                  setIsDebugging(!isDebugging);
                }
                }>
                {isDebugging ?
                  <BugAntIcon
                    className="w-3 text-gray-500"
                    aria-hidden="true" />
                  :
                  <BugAntIconOutline
                    className="w-3 text-gray-500"
                    aria-hidden="true" />}
              </button>
            </div>
          </div>
          {/* button to run the algorithm */}
          <button type="button"
            className="rounded-md border-0 text-white bg-green-500 px-3 py-2 text-sm font-semibold shadow-md hover:bg-gray-400" onClick={async () => {
              setIsProcessing(true);
              await computeDefects(img1URL, img2URL, method, resolution);
              setIsProcessing(false);
            }}>
            {/* loading indicator */}
            <span className='flex flex-col justify-center'>
              {isProcessing ?
                <svg className="block animate-spin h-5 w-5 text-white mx-auto" xmlns="http://www.w3.org/2000/svg" fill="none"
                  viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
                </svg>
                :
                <PlayIcon className="block h-5 w-5 text-white mx-auto" aria-hidden="true"></PlayIcon>
              }
              <span>Compare</span>
            </span>
          </button>
        </div>
      </div>

      <div className="bg-gray-50">

        <main className="mx-auto pb-48 pt-4 px-4">

          {/* Products */}
          <div className="flex flex-row my-1 space-x-4">
            <div className="flex flex-1 bg-white shadow p-4 rounded">
              <div className="flex-1 flex flex-col">
                <div className="text-lg font-semibold text-gray-900">Clean</div>
                <div className="text-sm font-semibold text-gray-900"># abnormal pixels: {numPixelsNegative}</div>
                <div className="text-sm font-semibold text-gray-900"># abnormal boxes: {numBoxesNegative}
                  <br></br><span className="font-normal">(click on the box to remove it)</span>
                </div>
              </div>
              <div className="aspect-w-1 aspect-h-1 ml-4">
                <img className="aspect-content w-24" src={placeholder(img1URL)} alt="Clean Image"></img>
              </div>
            </div>
            <div className="flex flex-1 bg-white shadow p-4 rounded">
              <div className="flex-1 flex flex-col">
                <div className="text-lg font-semibold text-gray-900">Defective</div>
                <div className="text-sm font-semibold text-gray-900"># abnormal pixels: {numPixels}</div>
                <div className="text-sm font-semibold text-gray-900"># abnormal boxes: {numBoxes}
                  <br></br>
                  <span className="font-normal">(click on the box to remove it)</span>
                </div>
              </div>
              <div className="aspect-w-1 aspect-h-1 ml-4">
                <img className="aspect-content w-24" src={placeholder(img2URL)} alt="Defective Image"></img>
              </div>
            </div>
          </div>

          {/* canvas for display, the canvas should be on top of one another */}
          <div className="relative flex flex-row mt-4">
            {/* download image from canvas button */}
            <button
              type="button"
              className="absolute top-0 right-0 bg-white rounded-bl px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50" onClick={() => {
                downloadImage(imageAbnormalOverlayRef);
              }
              }>
              <ArrowDownOnSquareIcon className="w-5 text-gray-500" aria-hidden="true"></ArrowDownOnSquareIcon>
            </button>
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
            <label>Difference (defective)</label>
            <canvas className="w-full" ref={imageAbnormalRef} id="imageAbnormal" width="300" height="300"></canvas>
            <label>Difference (clean)</label>
            <canvas className="w-full" ref={imageAbnormalNegRef} id="imageAbnormalNeg" width="300" height="300"></canvas>
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
