import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Filter, CheckCircle2, XCircle } from 'lucide-react';
import { motion } from 'framer-motion';

const featureSelectionData = [
  { 
    feature: 'Systolic BP', 
    rfe: true, 
    chiSquare: 45.2, 
    mutualInfo: 0.38, 
    genetic: true, 
    pso: true, 
    xgboost: 0.42 
  },
  { 
    feature: 'Blood Sugar', 
    rfe: true, 
    chiSquare: 38.7, 
    mutualInfo: 0.31, 
    genetic: true, 
    pso: true, 
    xgboost: 0.35 
  },
  { 
    feature: 'Age', 
    rfe: true, 
    chiSquare: 32.1, 
    mutualInfo: 0.28, 
    genetic: true, 
    pso: false, 
    xgboost: 0.28 
  },
  { 
    feature: 'Heart Rate', 
    rfe: true, 
    chiSquare: 28.5, 
    mutualInfo: 0.24, 
    genetic: false, 
    pso: true, 
    xgboost: 0.22 
  },
  { 
    feature: 'Diastolic BP', 
    rfe: false, 
    chiSquare: 22.3, 
    mutualInfo: 0.19, 
    genetic: true, 
    pso: true, 
    xgboost: 0.18 
  },
  { 
    feature: 'Body Temp', 
    rfe: false, 
    chiSquare: 15.8, 
    mutualInfo: 0.12, 
    genetic: false, 
    pso: false, 
    xgboost: 0.12 
  },
];

const techniques = [
  { key: 'rfe', label: 'RFE', description: 'Recursive Feature Elimination', explanation: 'Iteratively removes least important features and rebuilds model until optimal subset is found. Helps identify the minimal set of vitals needed for accurate prediction.', color: 'violet' },
  { key: 'chiSquare', label: 'Chi-Square', description: 'Statistical dependency test', explanation: 'Measures statistical dependence between each feature and risk outcome. Higher scores indicate stronger association with maternal risk levels.', color: 'blue' },
  { key: 'mutualInfo', label: 'Mutual Info', description: 'Information gain measure', explanation: 'Quantifies how much knowing a feature reduces uncertainty about the risk level. Captures both linear and non-linear relationships.', color: 'emerald' },
  { key: 'genetic', label: 'Genetic Algo', description: 'Evolutionary optimization', explanation: 'Uses natural selection principles to evolve optimal feature combinations. Explores diverse feature subsets to find best predictors.', color: 'amber' },
  { key: 'pso', label: 'PSO', description: 'Particle Swarm Optimization', explanation: 'Simulates swarm behavior to search for optimal features. Particles collaborate to discover which vitals contribute most to accurate predictions.', color: 'rose' },
  { key: 'xgboost', label: 'XGBoost', description: 'Embedded importance', explanation: 'Built-in feature ranking from gradient boosting trees. Shows how often each feature is used in decision splits during model training.', color: 'cyan' },
];

export default function FeatureSelection() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <Card className="bg-slate-800/50 backdrop-blur-xl border border-white/10">
        <CardHeader className="border-b border-white/10">
          <CardTitle className="flex items-center gap-3 text-white">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center">
              <Filter className="w-5 h-5 text-white" />
            </div>
            Feature Selection Techniques
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          {/* Technique Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
            {techniques.map((tech, index) => (
              <motion.div
                key={tech.key}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.05 }}
                className="bg-slate-700/30 border border-slate-600/30 rounded-xl p-4 hover:border-teal-500/30 transition-colors"
              >
                <div className="flex items-center gap-2 mb-2">
                  <div className={`w-2 h-2 rounded-full ${
                    tech.color === 'violet' ? 'bg-violet-400' :
                    tech.color === 'blue' ? 'bg-blue-400' :
                    tech.color === 'emerald' ? 'bg-emerald-400' :
                    tech.color === 'amber' ? 'bg-amber-400' :
                    tech.color === 'rose' ? 'bg-rose-400' :
                    'bg-cyan-400'
                  }`} />
                  <p className="font-semibold text-white text-sm">{tech.label}</p>
                </div>
                <p className="text-teal-400 text-xs font-medium mb-1">{tech.description}</p>
                <p className="text-slate-400 text-xs leading-relaxed">{tech.explanation}</p>
              </motion.div>
            ))}
          </div>

          {/* Comparison Table */}
          <div className="rounded-xl border border-slate-600/30 overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow className="bg-slate-700/50 border-slate-600/30">
                  <TableHead className="text-slate-300">Feature</TableHead>
                  <TableHead className="text-slate-300 text-center">RFE</TableHead>
                  <TableHead className="text-slate-300 text-center">Chi²</TableHead>
                  <TableHead className="text-slate-300 text-center">MI</TableHead>
                  <TableHead className="text-slate-300 text-center">GA</TableHead>
                  <TableHead className="text-slate-300 text-center">PSO</TableHead>
                  <TableHead className="text-slate-300 text-center">XGB</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {featureSelectionData.map((row, index) => (
                  <TableRow key={row.feature} className="border-slate-600/30 hover:bg-slate-700/30">
                    <TableCell className="font-medium text-white">{row.feature}</TableCell>
                    <TableCell className="text-center">
                      {row.rfe ? (
                        <CheckCircle2 className="w-5 h-5 text-emerald-400 mx-auto" />
                      ) : (
                        <XCircle className="w-5 h-5 text-slate-500 mx-auto" />
                      )}
                    </TableCell>
                    <TableCell className="text-center">
                      <Badge variant="outline" className="bg-blue-500/10 text-blue-400 border-blue-500/30">
                        {row.chiSquare}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-center">
                      <Badge variant="outline" className="bg-emerald-500/10 text-emerald-400 border-emerald-500/30">
                        {row.mutualInfo}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-center">
                      {row.genetic ? (
                        <CheckCircle2 className="w-5 h-5 text-emerald-400 mx-auto" />
                      ) : (
                        <XCircle className="w-5 h-5 text-slate-500 mx-auto" />
                      )}
                    </TableCell>
                    <TableCell className="text-center">
                      {row.pso ? (
                        <CheckCircle2 className="w-5 h-5 text-emerald-400 mx-auto" />
                      ) : (
                        <XCircle className="w-5 h-5 text-slate-500 mx-auto" />
                      )}
                    </TableCell>
                    <TableCell className="text-center">
                      <Badge variant="outline" className="bg-cyan-500/10 text-cyan-400 border-cyan-500/30">
                        {row.xgboost}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}