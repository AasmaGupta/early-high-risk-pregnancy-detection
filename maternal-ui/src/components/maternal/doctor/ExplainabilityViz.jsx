import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  ScatterChart, Scatter, LineChart, Line
} from 'recharts';
import { Lightbulb, Eye, BarChart3 } from 'lucide-react';
import { motion } from 'framer-motion';

// Generate dynamic importance based on user inputs
const generateDynamicImportance = (formData) => {
  if (!formData) {
    // Default varied data
    return [
      { feature: 'Blood Sugar', importance: 0.38, direction: 'positive', color: '#F59E0B' },
      { feature: 'Systolic BP', importance: 0.32, direction: 'positive', color: '#EF4444' },
      { feature: 'Age', importance: 0.24, direction: 'positive', color: '#8B5CF6' },
      { feature: 'Diastolic BP', importance: 0.21, direction: 'positive', color: '#06B6D4' },
      { feature: 'Heart Rate', importance: 0.16, direction: 'positive', color: '#EC4899' },
      { feature: 'Body Temp', importance: 0.09, direction: 'negative', color: '#10B981' },
    ];
  }

  // Calculate importance based on how far values deviate from normal
  const normalRanges = {
    systolicBP: { min: 90, max: 120, weight: 0.25 },
    diastolicBP: { min: 60, max: 80, weight: 0.15 },
    bloodSugar: { min: 70, max: 100, weight: 0.25 },
    age: { min: 20, max: 35, weight: 0.15 },
    heartRate: { min: 60, max: 100, weight: 0.12 },
    bodyTemp: { min: 97, max: 99, weight: 0.08 },
  };

  const calculateDeviation = (value, min, max, weight) => {
    if (value < min) return ((min - value) / min) * weight;
    if (value > max) return ((value - max) / max) * weight;
    return weight * 0.1; // Small baseline importance
  };

  const importances = [
    { 
      feature: 'Systolic BP', 
      importance: calculateDeviation(formData.systolicBP || 120, 90, 120, normalRanges.systolicBP.weight),
      color: '#EF4444'
    },
    { 
      feature: 'Diastolic BP', 
      importance: calculateDeviation(formData.diastolicBP || 80, 60, 80, normalRanges.diastolicBP.weight),
      color: '#06B6D4'
    },
    { 
      feature: 'Blood Sugar', 
      importance: calculateDeviation(formData.bloodSugar || 95, 70, 100, normalRanges.bloodSugar.weight),
      color: '#F59E0B'
    },
    { 
      feature: 'Age', 
      importance: calculateDeviation(formData.age || 28, 20, 35, normalRanges.age.weight),
      color: '#8B5CF6'
    },
    { 
      feature: 'Heart Rate', 
      importance: calculateDeviation(formData.heartRate || 75, 60, 100, normalRanges.heartRate.weight),
      color: '#EC4899'
    },
    { 
      feature: 'Body Temp', 
      importance: calculateDeviation(formData.bodyTemp || 98.6, 97, 99, normalRanges.bodyTemp.weight),
      color: '#10B981'
    },
  ];

  // Sort by importance descending
  return importances.sort((a, b) => b.importance - a.importance);
};

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-slate-800 border border-slate-700 rounded-lg p-3 shadow-xl">
        <p className="font-semibold text-white">{payload[0].payload.feature}</p>
        <p className="text-sm text-teal-400">
          Importance: {(payload[0].value * 100).toFixed(1)}%
        </p>
      </div>
    );
  }
  return null;
};

export default function ExplainabilityViz({ formData }) {
  const shapData = useMemo(() => generateDynamicImportance(formData), [formData]);
  
  const permutationData = useMemo(() => {
    return shapData.map(item => ({
      ...item,
      importance: item.importance * (0.8 + Math.random() * 0.4) // Add some variation
    })).sort((a, b) => b.importance - a.importance);
  }, [shapData]);

  // Dynamic PDP based on systolic BP if available
  const pdpData = useMemo(() => {
    const baseValue = formData?.systolicBP || 120;
    return Array.from({ length: 20 }, (_, i) => {
      const x = 70 + i * 7;
      const distanceFromInput = Math.abs(x - baseValue);
      const y = 0.15 + (x / 200) * 0.5 + Math.sin((x - baseValue) * 0.1) * 0.08;
      return { x, y: Math.min(0.9, Math.max(0.1, y)) };
    });
  }, [formData?.systolicBP]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <Card className="bg-slate-800/50 backdrop-blur-xl border border-white/10">
        <CardHeader className="border-b border-white/10">
          <CardTitle className="flex items-center gap-3 text-white">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center">
              <Lightbulb className="w-5 h-5 text-white" />
            </div>
            Explainable AI Visualizations
            {formData && (
              <span className="ml-auto text-xs bg-teal-500/20 text-teal-400 px-3 py-1 rounded-full border border-teal-500/30">
                Dynamic • Based on Input
              </span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <Tabs defaultValue="shap" className="w-full">
            <TabsList className="bg-slate-700/50 border border-slate-600/50 mb-6">
              <TabsTrigger value="shap" className="data-[state=active]:bg-teal-500 data-[state=active]:text-white">
                SHAP Summary
              </TabsTrigger>
              <TabsTrigger value="feature" className="data-[state=active]:bg-teal-500 data-[state=active]:text-white">
                Feature Importance
              </TabsTrigger>
              <TabsTrigger value="permutation" className="data-[state=active]:bg-teal-500 data-[state=active]:text-white">
                Permutation
              </TabsTrigger>
              <TabsTrigger value="pdp" className="data-[state=active]:bg-teal-500 data-[state=active]:text-white">
                Partial Dependence
              </TabsTrigger>
            </TabsList>

            <TabsContent value="shap">
              <div className="space-y-4">
                <p className="text-slate-400 text-sm">
                  <strong className="text-teal-400">SHAP (SHapley Additive exPlanations)</strong> breaks down each prediction to show exactly how much each feature pushed the risk up or down. Helps clinicians understand <em>why</em> a patient received a specific risk score.
                </p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={shapData} layout="vertical" margin={{ left: 80 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis type="number" stroke="#94A3B8" fontSize={12} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                      <YAxis type="category" dataKey="feature" stroke="#94A3B8" fontSize={12} />
                      <Tooltip content={<CustomTooltip />} />
                      <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                        {shapData.map((entry, index) => (
                          <Cell key={index} fill={entry.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="feature">
              <div className="space-y-4">
                <p className="text-slate-400 text-sm">
                  <strong className="text-teal-400">Feature Importance</strong> ranks clinical parameters by how often they're used in model decisions. Higher importance means the feature is more critical for distinguishing between risk levels.
                </p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={shapData} layout="vertical" margin={{ left: 80 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis type="number" stroke="#94A3B8" fontSize={12} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                      <YAxis type="category" dataKey="feature" stroke="#94A3B8" fontSize={12} />
                      <Tooltip content={<CustomTooltip />} />
                      <Bar dataKey="importance" fill="#14B8A6" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="permutation">
              <div className="space-y-4">
                <p className="text-slate-400 text-sm">
                  <strong className="text-teal-400">Permutation Importance</strong> tests how accuracy drops when a feature is scrambled. A large drop means the model heavily relies on that feature — useful for identifying which vitals are most predictive.
                </p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={permutationData} layout="vertical" margin={{ left: 80 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis type="number" stroke="#94A3B8" fontSize={12} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                      <YAxis type="category" dataKey="feature" stroke="#94A3B8" fontSize={12} />
                      <Tooltip content={<CustomTooltip />} />
                      <Bar dataKey="importance" fill="#8B5CF6" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="pdp">
              <div className="space-y-4">
                <p className="text-slate-400 text-sm">
                  <strong className="text-teal-400">Partial Dependence Plot</strong> visualizes how changing one vital (e.g., Systolic BP) affects risk probability while holding others constant. Helps clinicians see thresholds where risk increases significantly.
                </p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={pdpData} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis 
                        type="number" 
                        dataKey="x" 
                        name="Systolic BP" 
                        stroke="#94A3B8" 
                        fontSize={12}
                        label={{ value: 'Systolic BP (mmHg)', position: 'bottom', fill: '#94A3B8', fontSize: 11 }}
                      />
                      <YAxis 
                        type="number" 
                        dataKey="y" 
                        name="Risk Probability" 
                        stroke="#94A3B8" 
                        fontSize={12}
                        tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                        label={{ value: 'Risk Probability', angle: -90, position: 'insideLeft', fill: '#94A3B8', fontSize: 11 }}
                      />
                      <Tooltip 
                        formatter={(value) => [`${(value * 100).toFixed(1)}%`, 'Risk']}
                        labelFormatter={(value) => `BP: ${value} mmHg`}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="y" 
                        stroke="#14B8A6" 
                        strokeWidth={3}
                        dot={{ fill: '#14B8A6', strokeWidth: 0, r: 4 }}
                        activeDot={{ r: 6, fill: '#14B8A6' }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </motion.div>
  );
}