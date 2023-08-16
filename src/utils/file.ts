import { Subject } from "rxjs";
import { files, images, DFImage, DFImageFile } from "../db/db";

export async function loadImageElementFromURL(imageUrl) {
  const image = await new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.src = imageUrl;
    image.onload = () => {
      resolve(image);
    };

    image.onerror = (error) => {
      reject(error);
    };
  });
  return image;
}

export async function loadImage(file) {
  const arrayBuffer: ArrayBuffer = await readFile(file);

  const arrayBufferView = new Uint8Array(arrayBuffer);
  const blob = new Blob([arrayBufferView]);
  const urlCreator = window.URL || window.webkitURL;

  const imageUrl = urlCreator.createObjectURL(blob);

  const image = await new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.src = imageUrl;
    image.onload = () => {
      resolve(image);
      urlCreator.revokeObjectURL(imageUrl);
    };

    image.onerror = (error) => {
      urlCreator.revokeObjectURL(imageUrl);
      reject(error);
    };
  });

  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  // context.imageSmoothingEnabled = false;
  const width = image.naturalWidth;
  const height = image.naturalHeight;
  canvas.height = image.naturalHeight;
  canvas.width = image.naturalWidth;
  context.drawImage(image, 0, 0);
  const dataBlob: Blob = await toBlob(canvas);

  return [width, height, dataBlob];
}

export async function readFile(file) {
  const subject = new Subject<ArrayBuffer>();
  let reader = new FileReader();
  reader.onload = function () {
    subject.next(this.result as ArrayBuffer);
    subject.complete();
  };
  reader.readAsArrayBuffer(file);
  return subject.toPromise();
}

export const importFile = async (file, index) => {
  if ((await files.where("name").equals(file.name).count()) > 0) {
    return;
  }
  console.log("file.type", file.type);
  let width, height, dataBlob;
  if (file.type.startsWith("image")) {
    let [_widthImg, _heightImg, _dataBlobImg] = await loadImage(file);
    width = _widthImg;
    height = _heightImg;
    dataBlob = _dataBlobImg;
  } else {
    // file not supported
    return;
  }
  const image: DFImage = {
    buffer: await dataBlob.arrayBuffer(),
    width: width,
    height: height,
    size: dataBlob.size,
    type: dataBlob.type,
  };
  const imageId = await images.put(image);

  const data: DFImageFile = {
    lastModified: file.lastModified,
    width: width,
    height: height,
    name: file.name,
    size: file.size,
    type: file.type,
    imageId: +imageId,
    index: index,
  };
  const fileId = await files.add(data);

  return fileId;
};

export async function toBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  const subject = new Subject<Blob>();
  canvas.toBlob(function (blob) {
    subject.next(blob);
    subject.complete();
  }, "image/jpeg");
  return subject.toPromise();
}
