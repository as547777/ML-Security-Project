import React from 'react';
import {useData} from "@/context/DataContext";
import {Menu} from "lucide-react";

const DefenseDetails = ({clickable, selectable} : {clickable?: boolean, selectable?: boolean}) => {
  const { defense } = useData()

  return (
    <div className={`rounded-xl bg-gradient-to-br p-5 text-white drop-shadow-md transition
                    ${clickable && "cursor-pointer hover:shadow-xl hover:from-green-700 hover:to-emerald-700"} 
                    ${!selectable ? "from-gray-600 to-stone-600" : "from-green-600 to-emerald-600"}`}>
      <div className="flex items-start justify-between">
        <div>
          <div className="text-xs font-medium mb-1 flex gap-1">
            {clickable && <Menu size={15} />} <span>Defense</span>
          </div>

          <h2 className="text-2xl font-bold">
            {selectable ? defense?.display_name : "Select a dataset first"}
          </h2>

          {defense && selectable ? (
            <>
              <p className="text-sm mt-2">
                {defense?.description || ''}
              </p>

              {(defense?.type || defense?.time) && (
                <div className="flex gap-4 text-sm mt-3">
                  {defense?.type && (
                    <div>
                      <span>Type:</span>{" "}
                      <span className="font-semibold">{defense.type}</span>
                    </div>
                  )}

                  {defense?.time && (
                    <div>
                      <span>Time:</span>{" "}
                      <span className="font-semibold">{defense.time}</span>
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

export default DefenseDetails;