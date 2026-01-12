import React from 'react';
import {useData} from "@/context/DataContext";
import {Menu} from "lucide-react";

const ModelDetails = ({clickable, selectable} : {clickable?: boolean, selectable?: boolean}) => {
  const { modelFamily } = useData()

  const nameText = selectable ? "Not selected" : "Select a dataset first"

  return (
    <div className={`rounded-xl bg-gradient-to-br p-5 text-white drop-shadow-md transition
                    ${clickable && "cursor-pointer hover:shadow-xl hover:from-indigo-700 hover:to-emerald-700"}
                    ${!selectable ? "from-gray-600 to-stone-600" : "from-indigo-600 to-emerald-600"}`}>
      <div className="flex items-start justify-between">
        <div>
          <div className="text-xs font-medium mb-1 flex gap-1">
            {clickable && <Menu size={15} />} <span>Model Family</span>
          </div>

          <h2 className="text-2xl font-bold">
            {modelFamily?.name || nameText}
          </h2>

          {modelFamily ? (
            <>
              <p className="text-sm mt-2 mb-3">
                {modelFamily?.description || ''}
              </p>

              <div className="flex gap-4 text-sm">
                <div>
                  <span>Category:</span>{" "}
                  <span className="font-semibold">{modelFamily?.category || ''}</span>
                </div>

                <div>
                  <span>Use case:</span>{" "}
                  <span className="font-semibold">{modelFamily?.use_case || ''}</span>
                </div>
              </div>
            </>
          ):(
            <></>
          )}
        </div>
      </div>
    </div>
  );
};

export default ModelDetails;