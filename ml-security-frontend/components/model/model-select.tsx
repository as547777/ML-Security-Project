'use client'

import React, {useState} from 'react';
import ModelDetails from "@/components/model/model-details";
import FieldInput from "@/components/field-input";

const ModelSelect = () => {
  const [ currModel, setCurrModel ] = useState('ResNet50')

  return (
    <div>
      <ModelDetails clickable />

      <div className={'block mt-4'}>
        <FieldInput
          label={'Model'}
          tooltip={'Choose a specific model from the selected family'}
          type={'select'}
          options={['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']}
          value={currModel}
          setValue={setCurrModel} />
      </div>
    </div>
  );
};

export default ModelSelect;