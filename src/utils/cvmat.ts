import cv, { } from "@techstark/opencv-js";

export function matToJson(mat) {
  // TODO: compress mat.data before storing
  let obj = {
    data: mat.data,
    rows: mat.rows,
    cols: mat.cols,
    type: mat.type(),
  };
  return obj;
}

export function jsonToMat(obj) {
  if (!obj) return null;
  let mat = new cv.Mat(obj.rows, obj.cols, obj.type);
  // TODO: decompress mat.data after retrieving
  mat.data.set(obj.data);
  return mat;
}
