import { useEffect, useState } from 'react';
import { pairs, pairData, DFPairData, DFPair } from '../db/db';
import { ImageItem } from './PairList';
export const Report = () => {
  // const [pairsState, setPairsState] = useState<DFPair[]>([])
  const [pairDataState, setPairDataState] = useState<DFPairData[]>([])

  useEffect(() => {
    setTimeout(async () => {
      // const pairsState = await pairs.toArray();
      // setPairsState(pairsState)
      const pairDataState = await pairData.toArray();
      setPairDataState(pairDataState)
    })
  }, [])

  // generate table report, first column is clean image, second column is defective image, third column is abnormal boxes
  return (<>
    {/* table */}
    <table className="table-auto">
      <thead>
        <tr>
          <th>Clean Image</th>
          <th>Defective Image</th>
          <th>Abnormal Boxes</th>
        </tr>
      </thead>
      <tbody>
        {pairDataState.map((pairData) => {          
          return (
            <>
              <tr key={pairData.id}>
                <td>
                </td>
                <td>
                </td>
              </tr>
            </>)
        })}
      </tbody>
    </table>
  </>)
}