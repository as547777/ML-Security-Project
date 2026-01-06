import React from 'react';
import {useData} from "@/context/DataContext";

const DefenseDetails = ({clickable} : {clickable?: boolean}) => {
  const { defense } = useData()

  return (
    <div className={`rounded-xl bg-gradient-to-br from-green-600 to-emerald-600 p-5 text-white shadow-sm transition
                    ${clickable ? "cursor-pointer hover:shadow-xl hover:from-green-700 hover:to-emerald-700" : ""}`}>
      <div className="flex items-start justify-between">
        <div>
          <div className="text-xs font-medium text-purple-100 mb-1">
            Defense
          </div>

          <h2 className="text-2xl font-bold mb-2">
            {defense?.name || "Not selected"}
          </h2>

          <p className="text-sm text-purple-100 mb-3">
            {defense?.description || ''}
          </p>

          <div className="flex gap-4 text-sm">
            <div>
              <span className="text-purple-200">Type:</span>{" "}
              <span className="font-semibold">{defense?.type || ''}</span>
            </div>

            <div>
              <span className="text-purple-200">nesto...:</span>{" "}
              <span className="font-semibold">{''}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DefenseDetails;