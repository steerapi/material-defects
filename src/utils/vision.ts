import { MutableRefObject } from "react";
import cv, { } from "@techstark/opencv-js";

export function computeCDF(hist) {
  let cdf = new Array(256).fill(0);
  cdf[0] = hist.data32F[0];
  for (let i = 1; i < 256; i++) {
    cdf[i] = cdf[i - 1] + hist.data32F[i];
  }
  return cdf;
}

export function histogramMatching(src, reference, dst) {
  let srcHist = new cv.Mat(),
    refHist = new cv.Mat();
  let mask = new cv.Mat();
  let dsize = new cv.Size(256, 1);
  let ranges = [0, 256];
  let histSize = [256];

  let srcMatVec = new cv.MatVector();
  let refMatVec = new cv.MatVector();
  srcMatVec.push_back(src);
  refMatVec.push_back(reference);
  cv.calcHist(srcMatVec, [0], mask, srcHist, histSize, ranges);
  cv.calcHist(refMatVec, [0], mask, refHist, histSize, ranges);

  let srcCDF = computeCDF(srcHist);
  let refCDF = computeCDF(refHist);

  let lut = new Array(256).fill(0);
  for (let i = 0; i < 256; i++) {
    let diff = Array(256).fill(255);
    for (let j = 0; j < 256; j++) {
      diff[j] = Math.abs(srcCDF[i] - refCDF[j]);
    }
    lut[i] = diff.indexOf(Math.min(...diff));
  }

  for (let i = 0; i < src.rows; i++) {
    for (let j = 0; j < src.cols; j++) {
      dst.data[i * src.cols + j] = lut[src.data[i * src.cols + j]];
    }
  }

  srcHist.delete();
  refHist.delete();
  mask.delete();
  srcMatVec.delete();
  refMatVec.delete();
}

export function autoThreshold(image_abnormal, ratio = 0.8) {
  let minMax: any = (cv as any).minMaxLoc(image_abnormal);
  let threshold = Math.floor((minMax.maxVal - minMax.minVal) * ratio);
  return threshold;
}

export function mergeBboxes(bboxes: any[]) {
  let merged = true;
  while (merged) {
    merged = false;
    for (let i = 0; i < bboxes.length; i++) {
      let bbox1 = bboxes[i];
      for (let j = i + 1; j < bboxes.length; j++) {
        let bbox2 = bboxes[j];
        // if bbox1 and bbox2 overlap
        if (
          bbox1[0] < bbox2[2] &&
          bbox1[2] > bbox2[0] &&
          bbox1[1] < bbox2[3] &&
          bbox1[3] > bbox2[1]
        ) {
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

export function postProcessImageAbnormal(
  image_abnormal: cv.Mat,
  white_image_warped_gray: cv.Mat
) {
  // blur
  cv.GaussianBlur(
    image_abnormal,
    image_abnormal,
    new cv.Size(5, 5),
    0,
    0,
    cv.BORDER_DEFAULT
  );

  // let kernel = cv.Mat.ones(7, 7, cv.CV_8U);
  // dilation
  // cv.dilate(image_abnormal, image_abnormal, kernel, { x: -1, y: -1 }, 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
  // erosion
  // for (let i = 0; i < 2; i++)
  //   cv.erode(image_abnormal, image_abnormal, kernel, { x: -1, y: -1 }, 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
  // cv.dilate(image_abnormal, image_abnormal, kernel, { x: -1, y: -1 }, 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
  // kernel.delete();

  // remove edges from image_abnormal
  let white_image_warped_gray_not = new cv.Mat();
  cv.bitwise_not(white_image_warped_gray, white_image_warped_gray_not);
  cv.bitwise_and(image_abnormal, white_image_warped_gray_not, image_abnormal);
  white_image_warped_gray_not.delete();
}

export function downloadImage(
  imageAbnormalOverlayRef: MutableRefObject<HTMLCanvasElement>
) {
  // download image from canvas
  const canvas = imageAbnormalOverlayRef.current;
  const image = canvas
    .toDataURL("image/png", 1.0)
    .replace("image/png", "image/octet-stream");
  const link = document.createElement("a");
  // with time stamp in the name
  link.download = `image_${new Date().getTime()}.png`;
  link.href = image;
  link.click();
}
