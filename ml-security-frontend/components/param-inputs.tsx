import React from 'react';
import {ParamsType} from "@/types";
import FieldInput from "@/components/field-input";

interface Props {
  params: ParamsType | undefined
  updateParams: (key: keyof ParamsType, value: number | string) => void
}

const ParamInputs = ({params, updateParams} : Props) => {
  if (!params) {
    return null;
  }

  return (
    <div className="grid grid-cols-2 sm:grid-cols-2 gap-4">
      {Object.entries(params).map(([key, param]) => (
        <FieldInput
          key={key}
          label={param.label}
          tooltip={param.tooltip}
          type={param.type as "number" | "string"}
          step={param.step}
          options={param.options}
          value={param.value}
          setValue={(value) => {
            updateParams(key, value)
          }}
        />
      ))}
    </div>
  );
};

export default ParamInputs;