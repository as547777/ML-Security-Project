import React from 'react';
import DatasetTypeIcon from "@/components/dataset/dataset-type-icon";
import {useData} from "@/context/DataContext";
import {Menu} from "lucide-react";

const DatasetDetails = ({clickable} : {clickable?: boolean}) => {
  const { dataset } = useData()

  return (
    <div className={`rounded-xl bg-gradient-to-br from-purple-500 to-indigo-600 p-5 text-white drop-shadow-md transition
                    ${clickable ? "cursor-pointer hover:shadow-xl hover:from-purple-600 hover:to-indigo-700" : ""}`}>
      <div className="flex items-start justify-between">
        <div>
          <div className="text-xs font-medium mb-1 flex gap-1">
            {clickable && <Menu size={15} />} <span>Dataset</span>
          </div>
          <h2 className="text-2xl font-bold">{dataset?.display_name || "Not selected"}</h2>

          {dataset ? (
            <>
              <p className="text-sm mt-2 mb-3">{dataset?.description}</p>
              <div className="flex gap-4 text-sm">
                <div>
                  <span>Train:</span>{" "}
                  <span className="font-semibold">{dataset?.trainCount?.toLocaleString()}</span>
                </div>
                <div>
                  <span>Test:</span>{" "}
                  <span className="font-semibold">{dataset?.testCount?.toLocaleString()}</span>
                </div>
              </div>
            </>
          ):(
            <></>
          )}

        </div>
        <div className="text-5xl opacity-20">
          <DatasetTypeIcon type={dataset?.type} size={10} color={'text-white'} />
        </div>
      </div>
    </div>
  );
};

export default DatasetDetails;