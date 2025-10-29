import { motion } from 'framer-motion';
import React, {PropsWithChildren} from 'react';

const MainContainer = ({children} : PropsWithChildren) => {
  return (
    <motion.main
      className="flex flex-col items-center justify-center min-h-[calc(100vh-100px)] p-8"
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
    >
      {children}
    </motion.main>
  );
};

export default MainContainer;