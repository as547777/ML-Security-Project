"use client"
import {useData} from "@/context/DataContext";
import ParamInputs from "@/components/param-inputs";

const DefenseInputs = () => {
  const { defenseParams, updateDefenseParams } = useData()

  return <ParamInputs params={defenseParams} updateParams={updateDefenseParams} />
};

export default DefenseInputs;