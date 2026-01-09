import React from 'react';
import {useData} from "@/context/DataContext";

const AttackDetails = ({clickable, selectable} : {clickable?: boolean, selectable?: boolean}) => {
  const { attack } = useData()

  const nameText = selectable ? "Not selected" : "Select a dataset first"

  return (
    <div className={`rounded-xl bg-gradient-to-br from-red-600 to-orange-600 p-5 text-white shadow-sm transition
                    ${clickable ? "cursor-pointer hover:shadow-xl hover:from-red-700 hover:to-orange-700" : ""}`}>
      <div className="flex items-start justify-between">
        <div>
          <div className="text-xs font-medium text-purple-100 mb-1">
            Attack
          </div>

          <h2 className="text-2xl font-bold">
            {attack?.display_name || nameText}
          </h2>

          {attack ? (
            <>
              <p className="text-sm text-purple-100 mt-2 mb-3">
                {attack?.description || ''}
              </p>

              <div className="flex gap-4 text-sm">
                <div>
                  <span className="text-purple-200">Type:</span>{" "}
                  <span className="font-semibold">{attack?.type || ''}</span>
                </div>

                <div>
                  <span className="text-purple-200">Time:</span>{" "}
                  <span className="font-semibold">{attack?.time || ''}</span>
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

export default AttackDetails;