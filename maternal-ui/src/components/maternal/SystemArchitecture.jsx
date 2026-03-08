import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  User, 
  Settings, 
  Brain, 
  Lightbulb, 
  Shield,
  ArrowRight,
  Database,
  Cpu
} from 'lucide-react';
import { motion } from 'framer-motion';

const stages = [
  { 
    icon: User, 
    label: 'User Input', 
    description: 'Clinical parameters',
    color: 'from-violet-500 to-purple-500'
  },
  { 
    icon: Settings, 
    label: 'Preprocessing', 
    description: 'Normalization & validation',
    color: 'from-blue-500 to-cyan-500'
  },
  { 
    icon: Brain, 
    label: 'Ensemble ML', 
    description: 'Stacking + Soft Voting',
    color: 'from-teal-500 to-emerald-500'
  },
  { 
    icon: Lightbulb, 
    label: 'Explainability', 
    description: 'SHAP interpretation',
    color: 'from-amber-500 to-orange-500'
  },
  { 
    icon: Shield, 
    label: 'Risk Output', 
    description: 'Low / Mid / High',
    color: 'from-rose-500 to-pink-500'
  },
];

export default function SystemArchitecture() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <Card className="bg-slate-800/50 backdrop-blur-xl border border-white/10">
        <CardHeader className="border-b border-white/10">
          <CardTitle className="flex items-center gap-3 text-white">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-500 flex items-center justify-center">
              <Cpu className="w-5 h-5 text-white" />
            </div>
            System Architecture
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          {/* Desktop Flow */}
          <div className="hidden lg:flex items-center justify-between gap-4">
            {stages.map((stage, index) => (
              <React.Fragment key={stage.label}>
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex flex-col items-center text-center"
                >
                  <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${stage.color} flex items-center justify-center shadow-lg mb-3`}>
                    <stage.icon className="w-8 h-8 text-white" />
                  </div>
                  <p className="font-semibold text-white text-sm">{stage.label}</p>
                  <p className="text-xs text-slate-400 mt-1">{stage.description}</p>
                </motion.div>
                {index < stages.length - 1 && (
                  <ArrowRight className="w-6 h-6 text-slate-500 flex-shrink-0" />
                )}
              </React.Fragment>
            ))}
          </div>

          {/* Mobile Flow */}
          <div className="lg:hidden space-y-4">
            {stages.map((stage, index) => (
              <motion.div
                key={stage.label}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="flex items-center gap-4"
              >
                <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${stage.color} flex items-center justify-center shadow-lg flex-shrink-0`}>
                  <stage.icon className="w-6 h-6 text-white" />
                </div>
                <div className="flex-1">
                  <p className="font-semibold text-white text-sm">{stage.label}</p>
                  <p className="text-xs text-slate-400">{stage.description}</p>
                </div>
                {index < stages.length - 1 && (
                  <ArrowRight className="w-4 h-4 text-slate-500 rotate-90" />
                )}
              </motion.div>
            ))}
          </div>

          {/* Technical Details */}
          <div className="mt-8 grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="bg-slate-700/30 rounded-xl p-4 border border-slate-600/30">
              <Database className="w-6 h-6 text-blue-400 mb-2" />
              <p className="font-semibold text-white text-sm">Data Pipeline</p>
              <p className="text-xs text-slate-400 mt-1">
                Feature scaling, outlier detection, missing value imputation
              </p>
            </div>
            <div className="bg-slate-700/30 rounded-xl p-4 border border-slate-600/30">
              <Brain className="w-6 h-6 text-teal-400 mb-2" />
              <p className="font-semibold text-white text-sm">Model Ensemble</p>
              <p className="text-xs text-slate-400 mt-1">
                XGBoost, Random Forest, Gradient Boosting with meta-learner
              </p>
            </div>
            <div className="bg-slate-700/30 rounded-xl p-4 border border-slate-600/30">
              <Lightbulb className="w-6 h-6 text-amber-400 mb-2" />
              <p className="font-semibold text-white text-sm">Interpretability</p>
              <p className="text-xs text-slate-400 mt-1">
                SHAP values, feature importance, partial dependence plots
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}