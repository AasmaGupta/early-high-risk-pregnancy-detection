import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Code2 } from 'lucide-react';
import { motion } from 'framer-motion';

const technologies = [
  { name: 'Python', category: 'Backend', color: 'from-blue-500 to-yellow-500' },
  { name: 'Scikit-learn', category: 'ML', color: 'from-orange-500 to-amber-500' },
  { name: 'XGBoost', category: 'ML', color: 'from-emerald-500 to-green-500' },
  { name: 'Pandas', category: 'Data', color: 'from-indigo-500 to-purple-500' },
  { name: 'NumPy', category: 'Data', color: 'from-cyan-500 to-blue-500' },
  { name: 'SHAP', category: 'XAI', color: 'from-rose-500 to-pink-500' },
  { name: 'Matplotlib', category: 'Viz', color: 'from-teal-500 to-cyan-500' },
  { name: 'React', category: 'Frontend', color: 'from-sky-500 to-blue-500' },
  { name: 'Flask API', category: 'Backend', color: 'from-slate-500 to-gray-500' },
  { name: 'GitHub', category: 'DevOps', color: 'from-gray-600 to-slate-700' },
];

export default function TechStack() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <Card className="bg-slate-800/50 backdrop-blur-xl border border-white/10">
        <CardHeader className="border-b border-white/10">
          <CardTitle className="flex items-center gap-3 text-white">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center">
              <Code2 className="w-5 h-5 text-white" />
            </div>
            Technology Stack
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
            {technologies.map((tech, index) => (
              <motion.div
                key={tech.name}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.05 }}
                className="group relative"
              >
                <div className="bg-slate-700/30 rounded-xl p-4 border border-slate-600/30 hover:border-teal-500/50 transition-all duration-300 hover:bg-slate-700/50">
                  <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${tech.color} flex items-center justify-center mb-3 shadow-lg`}>
                    <span className="text-white font-bold text-sm">{tech.name[0]}</span>
                  </div>
                  <p className="font-semibold text-white text-sm">{tech.name}</p>
                  <p className="text-xs text-slate-400 mt-1">{tech.category}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}