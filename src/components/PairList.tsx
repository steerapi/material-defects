import { useEffect, useState } from "react"
import { DndProvider, useDrop } from "react-dnd"
import { DFPair, files, images, pairData, pairs } from "../db/db"
import { HTML5Backend } from "react-dnd-html5-backend"
import { TYPE } from "./FileManager"
import { useNavigate, useParams } from "react-router-dom"
import { classNames } from "../utils/react"
import { XCircleIcon } from "@heroicons/react/24/outline"
import { placeholder } from "../utils/image"
import { forEach } from "modern-async"

export function PairList(props) {
  const [pairsData, setPairsData] = useState<DFPair[]>([])
  const { pairId } = useParams();
  const navigate = useNavigate()

  useEffect(() => {
    setTimeout(async () => {
      const data = await pairs.toArray();
      setPairsData(data)
    })
  }, [])

  return (<>
    <DndProvider backend={HTML5Backend}>
      <div className="flex flex-row border-b-2">
        <div className="flex flex-col space-y-2 p-4 border-r-2">
          <button className={"rounded-md border-0 px-1 py-1 text-sm font-semibold shadow-md hover:bg-gray-400 bg-blue-500 text-white w-24 flex-grow-0 flex-shrink-0"} onClick={async () => {
            pairs.add({
              id: Date.now(),
            })
            const data = await pairs.toArray();
            setPairsData(data)
          }}>New Pair</button>
          {/* <button className={"rounded-md border-0 px-1 py-1 text-sm font-semibold shadow-md hover:bg-gray-400 bg-orange-500 text-white w-24 flex-grow-0 flex-shrink-0"} onClick={async () => {
            // navigate to new page in a new window
            window.open('/report', '_blank');
          }}>Report</button> */}
          <button className={"rounded-md border-0 px-1 py-1 text-sm font-semibold shadow-md hover:bg-gray-400 bg-red-500 text-white w-24 flex-grow-0 flex-shrink-0"} onClick={async () => {
            // clear images
            // get all images
            const allImages = await images.toArray();
            // get all files
            const allFiles = await files.toArray();
            // find images that are not in files imageId
            const imageIds = allFiles.map(file => file.imageId);
            const imagesToDelete = allImages.filter(image => {
              return !imageIds.includes(image.id);
            });
            // delete images
            await forEach(imagesToDelete, async (image) => {
              await images.delete(image.id);
            });

            await pairs.clear();
            const data = await pairs.toArray();
            setPairsData(data)
          }}>Clear</button>
        </div>

        <div className="flex flex-row overflow-x-scroll flex-nowrap">
          {pairsData.map((pair) => (
            <div key={pair.id} className="flex flex-row flex-shrink-0 flex-grow-0" onClick={() => {
              navigate('/' + pair.id);
            }}>
              <PairListItem pair={pair} onDelete={async () => {
                await pairs.delete(pair.id);
                await pairData.delete(pair.id);
                const data = await pairs.toArray();
                setPairsData(data)

                // if pairId == pair.id then navigate to first pair
                if (+pairId == pair.id) {
                  if (data.length > 0) {
                    navigate('/' + data[0].id);
                  } else {
                    navigate('/');
                  }
                }
              }
              }></PairListItem>
            </div>
          ))}
        </div>
      </div>
    </DndProvider>
  </>)
}

export function PairListItem({ pair: _pair, onDelete }) {
  const [pair, setPair] = useState(_pair)
  const { pairId } = useParams();
  const navigate = useNavigate()
  const [{ canDrop: canDrop1, isOver: isOver1 }, drop1] = useDrop(() => ({
    // The type (or types) to accept - strings or symbols
    accept: TYPE,
    // Props to collect
    collect: (monitor) => ({
      isOver: monitor.isOver(),
      canDrop: monitor.canDrop()
    }),
    drop: (droppedItem: { index: number, imageId: number, imageFileId: number }) => {
      // load image file into pair data
      pairs.update(pair.id, {
        cleanImageId: droppedItem.imageId,
        cleanImageFileId: droppedItem.imageFileId,
      })

      pair.cleanImageId = droppedItem.imageId
      pair.cleanImageFileId = droppedItem.imageFileId
      setPair({
        ...pair,
      })

      // if this is the first pair, auto navigate to this pair
      if (pair.cleanImageId && pair.defectiveImageId) {
        setTimeout(async () => {
          const count = await pairs.count();
          if (count == 1) {
            navigate('/' + pair.id);
          }
        })
      }
    },
  }))
  const [{ canDrop: canDrop2, isOver: isOver2 }, drop2] = useDrop(() => ({
    // The type (or types) to accept - strings or symbols
    accept: TYPE,
    // Props to collect
    collect: (monitor) => ({
      isOver: monitor.isOver(),
      canDrop: monitor.canDrop()
    }),
    drop: (droppedItem: { index: number, imageId: number, imageFileId: number }) => {
      // load image file into pair data
      pairs.update(pair.id, {
        defectiveImageId: droppedItem.imageId,
        defectiveImageFileId: droppedItem.imageFileId,
      })

      pair.defectiveImageId = droppedItem.imageId
      pair.defectiveImageFileId = droppedItem.imageFileId
      setPair({
        ...pair,
      })

      // if this is the first pair, auto navigate to this pair
      if (pair.cleanImageId && pair.defectiveImageId) {
        setTimeout(async () => {
          const count = await pairs.count();
          if (count == 1) {
            navigate('/' + pair.id);
          }
        })
      }
    },
  }))

  return (<>
    <div className="relative">
      {/* add border if pairId == pair.id */}
      <div className={classNames(pairId == pair.id && "border-red-400", "flex flex-row flex-shrink-0 flex-grow-0 p-0 m-2 border-2")}>
        <div ref={drop1} className="flex flex-col items-center p-1 border-gray-200">
          <label className="mb-1 block text-sm font-semibold text-gray-700">Clean</label>
          <div className="flex flex-row">
            {pair.cleanImageId ?
              <ImageItem imageId={pair.cleanImageId} /> :
              <div>
                <div className="text-xs h-20">{canDrop1 ? <ImageItem imageId={pair.cleanImageId} /> : 'Drag an image here'}</div>
              </div>
            }
          </div>
        </div>
        {/* vertical line separation */}
        <div className="border-r-2 border-gray-200"></div>
        <div ref={drop2} className="flex flex-col items-center p-1  border-gray-200">
          <label className="mb-1 block text-sm font-semibold text-gray-700">Defective</label>
          <div className="flex flex-row">
            {pair.defectiveImageId ?
              <ImageItem imageId={pair.defectiveImageId} /> :
              <div>
                <div className="text-xs">{canDrop2 ? <ImageItem imageId={pair.cleanImageId} /> : 'Drag an image here'}</div>
              </div>
            }
          </div>
        </div>
      </div>
      <button className="self-start rounded-md border-0 px-1 py-1 text-sm font-semibold shadow-md hover:bg-gray-400 bg-red-600 text-white absolute right-0 top-0" onClick={(event) => {
        event.stopPropagation();
        event.preventDefault();
        onDelete();
      }}>
        <XCircleIcon className="w-4 h-4" />
      </button>
    </div>
  </>)
}


export const ImageItem = ({ imageId }) => {
  const [imageUrl, setImageUrl] = useState("");
  const [name, setName] = useState("");

  useEffect(() => {
    setTimeout(async () => {
      if (!imageId) {
        return;
      }
      const imageData = await images.get(imageId);
      if (!imageData) {
        return;
      }
      const imageUrl = URL.createObjectURL(
        new Blob([imageData.buffer], { type: imageData.type })
      );
      setImageUrl(imageUrl);

      // find files with this imageId
      const file = await files.where('imageId').equals(imageId).first();
      console.log('file', file)
      if (file) {
        setName(file.name)
      }      
    })
  }, [imageId]);

  return (
    <>
      <div className="flex flex-row w-24 m-0 self-center relative">
        <img className="w-24 h-20 border-2 border-gray-400 rounded-md overflow-hidden" src={placeholder(imageUrl)} alt={imageId} />
        {/* name */}
        <div className="absolute bottom-0 left-0 right-0 bg-gray-800 bg-opacity-50 text-white text-xs px-1 py-1 whitespace-nowrap">{name}</div>
      </div>
    </>
  )
}