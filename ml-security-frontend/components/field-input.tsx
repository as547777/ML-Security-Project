import React from 'react';
import {Tooltip, TooltipContent, TooltipProvider, TooltipTrigger} from "@/components/ui/tooltip";
import {Info} from "lucide-react";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select";
import {useData} from "@/context/DataContext";

interface FieldInputProps<T extends string | number> {
  label: string;
  tooltip: string;
  type: 'number' | 'string' | 'select' | 'select_class';
  step?: number;
  options?: string[];
  value: T;
  setValue: ((value: T) => void);
}

const FieldInput = <T extends string | number>(params : FieldInputProps<T>) => {
  // const { dataset } = useData()
  //
  // const options = params.type === 'select_class'
  //   ? dataset?.classes
  //   : params.options;

  const field = () =>{
    // TODO - add min and max values for number inputs
    if (params.type === "number" || params.type === "string") {
      return (
        <input
          type={params.type}
          step={params.step}
          value={params.value}
          onChange={(e) => {
            const value =
              params.type === "number"
                ? (parseFloat(e.target.value) as T) : (e.target.value as T);
            params.setValue(value);
          }}
          className="param-input"
        />
      )
    } else if (params.type === "select" || params.type === "select_class") {
      // TODO - problem with parsing numbers here, defaultValue and value in SelectItem need to be the same type,
      //  might be smart to add   optionType?: 'number' | 'string';, and then do the following:
      //  onValueChange={(e) => {
      //   const value = params.optionType === 'number'
      //     ? (parseFloat(e) as T)
      //     : (e as T);
      //   params.setValue(value);
      //  }}
      return (
        <Select
          value={params.value}
          onValueChange={(e) => params.setValue(e as T)}
        >
          <SelectTrigger className="param-input">
            <SelectValue placeholder="Select" />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
                {params.options?.map((option) => (
                  <SelectItem key={option} value={option}>
                    {option}
                  </SelectItem>
                ))}
            </SelectGroup>
          </SelectContent>
        </Select>
      );
    }
  }

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

        {field()}
      </div>
    </TooltipProvider>
  );
};

export default FieldInput;