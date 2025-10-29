"use client"
import React from 'react';
import {useData} from "@/context/DataContext";

const DatasetSelect = () => {
  const { dataset, setDataset } = useData()

  return (
    <div>
      <label>
        Dataset
      </label>
      <select value={dataset}
              onChange={(e) => setDataset(e.target.value)}>
        <option>MNIST</option>
        <option>CIFAR-10</option>
        <option>Custom Dataset</option>
      </select>
    </div>
  );
};

export default DatasetSelect;