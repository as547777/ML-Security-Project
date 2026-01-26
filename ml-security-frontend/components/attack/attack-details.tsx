import React from 'react';
import {useData} from "@/context/DataContext";
import {Menu} from "lucide-react";

const AttackDetails = ({clickable, selectable} : {clickable?: boolean, selectable?: boolean}) => {
  const { attack } = useData()

  return (
    <div className={`rounded-xl bg-gradient-to-br p-5 text-white drop-shadow-md transition
                    ${clickable && "cursor-pointer hover:shadow-xl hover:from-red-700 hover:to-orange-700"}
                    ${!selectable ? "from-gray-600 to-stone-600" : "from-red-600 to-orange-600"}`}>
      <div className="flex items-start justify-between">
        <div>
          <div className="text-xs font-medium mb-1 flex gap-1">
            {clickable && <Menu size={15} />} <span>Attack</span>
          </div>

          <h2 className="text-2xl font-bold">
            {selectable ? attack?.display_name : "Select a dataset first"}
          </h2>

          {attack && selectable ? (
            <>
              <p className="text-sm mt-2">
                {attack?.description || ''}
              </p>

              {(attack?.type || attack?.time) && (
                <div className="flex gap-4 text-sm mt-3">
                  {attack?.type && (
                    <div>
                      <span>Type:</span>{" "}
                      <span className="font-semibold">{attack.type}</span>
                    </div>
                  )}

                  {attack?.time && (
                    <div>
                      <span>Time:</span>{" "}
                      <span className="font-semibold">{attack.time}</span>
                    </div>
                  )}
                </div>
              )}
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