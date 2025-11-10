import React from 'react';
import {Tooltip, TooltipContent, TooltipProvider, TooltipTrigger} from "@/components/ui/tooltip";
import {Info} from "lucide-react";

interface FieldInputProps<T extends string | number> {
  label: string;
  tooltip: string;
  type: 'number' | 'string'
  step?: number;
  value: T;
  setValue: ((value: T) => void);
}

const FieldInput = <T extends string | number>(params : FieldInputProps<T>) => {
  return (
    <TooltipProvider delayDuration={150}>
      <div className="param-container">
        <div className="flex w-full justify-between gap-1 mb-1">
          <label className="param-label">{params.label}</label>
          <Tooltip>
            <TooltipTrigger asChild>
              <Info
                size={14}
                className="text-zinc-400 hover:text-blue-500 transition-colors"
              />
            </TooltipTrigger>
            <TooltipContent className="max-w-xs text-sm text-zinc-200 pb-2">
              {params.tooltip}
            </TooltipContent>
          </Tooltip>
        </div>

        <input
          type={params.type}
          step={params.step}
          value={params.value}
          onChange={(e) => {
            const value =
              params.type === "number"
                ? (parseFloat(e.target.value) as T)
                : (e.target.value as T);
            params.setValue(value);
          }}
          className="param-input"
        />
      </div>
    </TooltipProvider>
  );
};

export default FieldInput;