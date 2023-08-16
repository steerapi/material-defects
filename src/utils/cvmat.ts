import cv, { } from "@techstark/opencv-js";

export function matToJson(mat) {
  console.log('matToJson', mat);
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
  mat.data.set(obj.data);
  return mat;
}
