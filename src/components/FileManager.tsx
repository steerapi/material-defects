import { useEffect, useRef, useState } from "react";
import { forEach } from "modern-async";
import { importFile } from "../utils/file";
import { DFImageFile, files, images } from "../db/db";
import { DndProvider, useDrag, useDrop } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { placeholder } from "../utils/image";

export const TYPE = 'IMAGE_ITEM';

export function FileManager() {
  const [loading, setLoading] = useState(false)
  const [imageFiles, setImageFiles] = useState<DFImageFile[]>([])
  const inpRef = useRef(null);

  const updateImageFiles = async () => {
    // load data    
    const imageFiles = await files.toArray();
    // sort by index
    imageFiles.sort((a, b) => {
      return a.index - b.index;
    });
    setImageFiles(imageFiles);
  }

  const selectFiles = async (event) => {
    console.log('event', event, event.target.files)
    setLoading(true);
    await forEach(event.target.files, async (file, index) => {
      await importFile(file, index);
    });
    updateImageFiles();

    setLoading(false);
  }

  useEffect(() => {
    (async () => {
      updateImageFiles();
    })()
  }, [])

  const moveImage = (fromIndex, toIndex) => {
    let newImageFiles = [...imageFiles];
    const [movedItem] = newImageFiles.splice(fromIndex, 1);
    newImageFiles.splice(toIndex, 0, movedItem);
    // update database files
    forEach(newImageFiles, async (imageFile, index) => {
      await files.update(imageFile.id, { index });
    });
    setImageFiles(newImageFiles);
  };

  return (
    <DndProvider backend={HTML5Backend}>
      <div className="flex flex-row border-b-2">
        <input ref={inpRef} className="w-40 hidden" accept="image/*" multiple type="file" onChange={(event) => {
          selectFiles(event);
        }} />
        <div className="flex flex-col space-y-2 p-4 border-r-2">
          <button className={"rounded-md border-0 px-1 py-1 text-sm font-semibold shadow-md hover:bg-gray-400 bg-blue-500 text-white w-24 flex-grow-0 flex-shrink-0"} onClick={() => {
            inpRef.current.click();
          }}>
            Add Files
          </button>
          <button className={"rounded-md border-0 px-1 py-1 text-sm font-semibold shadow-md hover:bg-gray-400 bg-red-500 text-white w-24 flex-grow-0 flex-shrink-0"} onClick={async () => {
            await files.clear();
            updateImageFiles();
          }}>Clear</button>
        </div>
        {/* show thumbnail of imageFiles */}
        <div className="flex flex-row space-x-2 w-full overflow-x-scroll flex-nowrap justify-start py-4 px-4">
          {imageFiles.map((imageFile, index) => {
            return <div
              key={imageFile.id}
              className="w-24 h-20 flex-grow-0 flex-shrink-0">
              <ImageItemDragDrop
                index={index}
                moveImage={moveImage}
                imageFile={imageFile}
                onDelete={async () => {
                  await files.delete(imageFile.id);
                  updateImageFiles();
                }}></ImageItemDragDrop>
            </div>
          })}
        </div>
      </div>
    </DndProvider>
  )
}

export const ImageItemDragDrop = ({ imageFile, onDelete, index, moveImage }) => {
  const [imageUrl, setImageUrl] = useState("");

  useEffect(() => {
    setTimeout(async () => {
      const imageData = await images.get(imageFile.imageId);
      if (!imageData) {
        return;
      }
      const imageUrl = URL.createObjectURL(
        new Blob([imageData.buffer], { type: imageData.type })
      );
      setImageUrl(imageUrl);
    })
  }, [imageFile]);

  const [, ref] = useDrag({
    type: TYPE,
    item: { index, imageFileId: imageFile.id, imageId: imageFile.imageId },
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
    }),
  });

  const [, drop] = useDrop({
    accept: TYPE,
    hover: (draggedItem: any) => {
      if (draggedItem.index !== index) {
        moveImage(draggedItem.index, index);
        draggedItem.index = index;
      }
    },
  });

  return (
    <>
      <div ref={(node) => ref(drop(node))}>
        <div className="flex flex-row w-24 m-0 self-center relative">
          <img className="w-24 h-20 border-2 border-gray-400 rounded-md overflow-hidden" src={placeholder(imageUrl)} alt={imageFile.name} />
          {/* name */}
          <div className="absolute bottom-0 left-0 right-0 bg-gray-800 bg-opacity-50 text-white text-xs px-1 py-1 whitespace-nowrap">{imageFile.name}</div>
          {/* delete imageFile button */}
          <button className="self-start rounded-md border-0 px-1 py-1 text-sm font-semibold shadow-md hover:bg-gray-400 bg-red-600 text-white absolute right-0 top-0" onClick={onDelete}>
            {/* close icon */}
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="w-3 h-3">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>
    </>
  )
}