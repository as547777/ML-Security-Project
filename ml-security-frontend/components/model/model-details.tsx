import React from 'react';

const ModelDetails = ({clickable} : {clickable?: boolean}) => {

  return (
    <div className={`rounded-xl bg-gradient-to-br from-indigo-600 to-emerald-600 p-5 text-white shadow-sm transition
                    ${clickable ? "cursor-pointer hover:shadow-xl hover:from-indigo-700 hover:to-emerald-700" : ""}`}>
      <div className="flex items-start justify-between">
        <div>
          <div className="text-xs font-medium text-purple-100 mb-1">
            Model family
          </div>

          <h2 className="text-2xl font-bold mb-2">
            ResNet
          </h2>

          <p className="text-sm text-purple-100 mb-3">
            Residual neural network architecture for deep image recognition tasks.
          </p>

          <div className="flex gap-4 text-sm">
            <div>
              <span className="text-purple-200">Category:</span>{" "}
              <span className="font-semibold">CNN</span>
            </div>

            <div>
              <span className="text-purple-200">Use case:</span>{" "}
              <span className="font-semibold">Computer Vision</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelDetails;