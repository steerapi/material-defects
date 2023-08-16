import { useEffect, useState } from "react"
import { DndProvider, useDrop } from "react-dnd"
import { DFPair, images, pairs } from "../db/db"
import { HTML5Backend } from "react-dnd-html5-backend"
import { TYPE } from "./FileManager"
import { useNavigate, useParams } from "react-router-dom"
import { classNames } from "../utils/react"
import { XCircleIcon } from "@heroicons/react/24/outline"

export function PairList(props) {
  const [pairsData, setPairsData] = useState<DFPair[]>([])
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
          <button className={"rounded-md border-0 px-1 py-1 text-sm font-semibold shadow-md hover:bg-gray-400 bg-orange-500 text-white w-24 flex-grow-0 flex-shrink-0"} onClick={async () => {
            
          }}>Report</button>
          <button className={"rounded-md border-0 px-1 py-1 text-sm font-semibold shadow-md hover:bg-gray-400 bg-red-500 text-white w-24 flex-grow-0 flex-shrink-0"} onClick={async () => {
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
                const data = await pairs.toArray();
                setPairsData(data)
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
                <div className="text-xs">{canDrop1 ? 'Release to place clean image' : 'Drag an image here'}</div>
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
                <div className="text-xs">{canDrop2 ? 'Release to place defective image' : 'Drag an image here'}</div>
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


const ImageItem = ({ imageId }) => {
  const [imageUrl, setImageUrl] = useState("");

  useEffect(() => {
    setTimeout(async () => {
      const imageData = await images.get(imageId);
      if (!imageData) {
        return;
      }
      const imageUrl = URL.createObjectURL(
        new Blob([imageData.buffer], { type: imageData.type })
      );
      setImageUrl(imageUrl);
    })
  }, [imageId]);

  return (
    <>
      <div className="self-center">
        <img className="w-24 h-20 border-2 border-gray-400 rounded-md overflow-hidden" src={imageUrl} alt={imageId} />
      </div>
    </>
  )
}