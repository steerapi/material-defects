import Dexie, { Table } from "dexie";

export interface DFImageFile {
  id?: number;
  name: string;
  imageId: number;
  lastModified: number;
  width: number;
  height: number;
  size: number;
  type: string;
  index: number;
}

export interface DFImage {
  id?: number;
  buffer: ArrayBuffer;
  width: number;
  height: number;
  size: number;
  type: string;
}

export interface DFPair {
  id?: number;
  cleanImageId?: number;
  defectiveImageId?: number;
  cleanImageFileId?: number;
  defectiveImageFileId?: number;
}

export class DefectDexie extends Dexie {
  // 'friends' is added by dexie when declaring the stores()
  // We just tell the typing system this is the case
  files!: Table<DFImageFile>;
  images!: Table<DFImage>;
  pairs!: Table<DFPair>;

  constructor() {
    super("defects");
    this.version(1).stores({
      files: "++id,&name",
      images: "++id",
      pairs: "++id",
    });
  }
}

export const db = new DefectDexie();
export const files = db.files;
export const images = db.images;
export const pairs = db.pairs;
