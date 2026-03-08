import React from 'react';
import { Activity, Shield, AlertTriangle } from 'lucide-react';
import { motion } from 'framer-motion';

export default function Header({ title, subtitle, showDisclaimer = true }) {
  return (
    <motion.div 
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="mb-8"
    >
      <div className="flex items-center gap-4 mb-2">
        <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-teal-400 to-cyan-500 flex items-center justify-center shadow-lg shadow-teal-500/20">
          <Activity className="w-7 h-7 text-white" />
        </div>
        <div>
          <h1 className="text-2xl lg:text-3xl font-bold text-white">{title}</h1>
          {subtitle && <p className="text-slate-400 mt-1">{subtitle}</p>}
        </div>
      </div>
      
      {showDisclaimer && (
        <div className="mt-4 flex items-start gap-3 bg-amber-500/10 border border-amber-500/20 rounded-xl p-4">
          <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-amber-200/80">
            <span className="font-semibold text-amber-400">Clinical Decision Support Tool</span> — 
            This system provides AI-assisted risk assessment and should not replace professional medical diagnosis or clinical judgment.
          </p>
        </div>
      )}
    </motion.div>
  );
}