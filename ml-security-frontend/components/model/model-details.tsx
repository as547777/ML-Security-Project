import React from 'react';
import {useData} from "@/context/DataContext";

const ModelDetails = ({clickable, selectable} : {clickable?: boolean, selectable?: boolean}) => {
  const { modelFamily } = useData()

  const nameText = selectable ? "Not selected" : "Select a dataset first"

  return (
    <div className={`rounded-xl bg-gradient-to-br from-indigo-600 to-emerald-600 p-5 text-white shadow-sm transition
                    ${clickable ? "cursor-pointer hover:shadow-xl hover:from-indigo-700 hover:to-emerald-700" : ""}`}>
      <div className="flex items-start justify-between">
        <div>
          <div className="text-xs font-medium text-purple-100 mb-1">
            Model family
          </div>

          <h2 className="text-2xl font-bold">
            {modelFamily?.name || nameText}
          </h2>

          {modelFamily ? (
            <>
              <p className="text-sm text-purple-100 mt-2 mb-3">
                {modelFamily?.description || ''}
              </p>

              <div className="flex gap-4 text-sm">
                <div>
                  <span className="text-purple-200">Category:</span>{" "}
                  <span className="font-semibold">{modelFamily?.category || ''}</span>
                </div>

                <div>
                  <span className="text-purple-200">Use case:</span>{" "}
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