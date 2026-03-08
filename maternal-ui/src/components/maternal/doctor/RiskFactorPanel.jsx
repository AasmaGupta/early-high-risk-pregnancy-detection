import React, { useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  AlertTriangle, 
  AlertCircle, 
  Shield,
  TrendingUp,
  TrendingDown,
  Minus
} from 'lucide-react';
import { motion } from 'framer-motion';

const analyzeFactors = (formData) => {
  if (!formData) return { high: [], moderate: [], protective: [] };

  const factors = {
    high: [],
    moderate: [],
    protective: [],
  };

  // Systolic BP
  if (formData.systolicBP > 140) {
    factors.high.push({ name: 'High Systolic BP', value: `${formData.systolicBP} mmHg`, impact: '+25%' });
  } else if (formData.systolicBP > 130) {
    factors.moderate.push({ name: 'Elevated Systolic BP', value: `${formData.systolicBP} mmHg`, impact: '+15%' });
  } else if (formData.systolicBP <= 120) {
    factors.protective.push({ name: 'Normal Systolic BP', value: `${formData.systolicBP} mmHg`, impact: '-5%' });
  }

  // Diastolic BP
  if (formData.diastolicBP > 90) {
    factors.high.push({ name: 'High Diastolic BP', value: `${formData.diastolicBP} mmHg`, impact: '+20%' });
  } else if (formData.diastolicBP > 85) {
    factors.moderate.push({ name: 'Elevated Diastolic BP', value: `${formData.diastolicBP} mmHg`, impact: '+10%' });
  } else if (formData.diastolicBP <= 80) {
    factors.protective.push({ name: 'Normal Diastolic BP', value: `${formData.diastolicBP} mmHg`, impact: '-3%' });
  }

  // Blood Sugar
  if (formData.bloodSugar > 140) {
    factors.high.push({ name: 'High Blood Sugar', value: `${formData.bloodSugar} mg/dL`, impact: '+25%' });
  } else if (formData.bloodSugar > 120) {
    factors.moderate.push({ name: 'Elevated Blood Sugar', value: `${formData.bloodSugar} mg/dL`, impact: '+15%' });
  } else if (formData.bloodSugar <= 100) {
    factors.protective.push({ name: 'Normal Blood Sugar', value: `${formData.bloodSugar} mg/dL`, impact: '-5%' });
  }

  // Age
  if (formData.age > 40) {
    factors.high.push({ name: 'Advanced Maternal Age', value: `${formData.age} years`, impact: '+15%' });
  } else if (formData.age > 35) {
    factors.moderate.push({ name: 'Elevated Maternal Age', value: `${formData.age} years`, impact: '+10%' });
  } else if (formData.age >= 20 && formData.age <= 30) {
    factors.protective.push({ name: 'Optimal Age Range', value: `${formData.age} years`, impact: '-5%' });
  }

  // Heart Rate
  if (formData.heartRate > 100) {
    factors.moderate.push({ name: 'Elevated Heart Rate', value: `${formData.heartRate} bpm`, impact: '+10%' });
  } else if (formData.heartRate >= 60 && formData.heartRate <= 80) {
    factors.protective.push({ name: 'Normal Heart Rate', value: `${formData.heartRate} bpm`, impact: '-2%' });
  }

  // Body Temperature
  if (formData.bodyTemp > 99.5) {
    factors.moderate.push({ name: 'Elevated Temperature', value: `${formData.bodyTemp}°F`, impact: '+5%' });
  } else if (formData.bodyTemp >= 97 && formData.bodyTemp <= 99) {
    factors.protective.push({ name: 'Normal Temperature', value: `${formData.bodyTemp}°F`, impact: '-1%' });
  }

  return factors;
};

export default function RiskFactorPanel({ formData }) {
  const factors = useMemo(() => analyzeFactors(formData), [formData]);

  const sections = [
    { 
      key: 'high', 
      title: 'High Impact Factors', 
      icon: AlertCircle, 
      color: 'from-rose-500 to-red-600',
      bgColor: 'bg-rose-500/10',
      borderColor: 'border-rose-500/30',
      textColor: 'text-rose-400',
      empty: 'No high impact factors detected'
    },
    { 
      key: 'moderate', 
      title: 'Moderate Impact Factors', 
      icon: AlertTriangle, 
      color: 'from-amber-500 to-orange-600',
      bgColor: 'bg-amber-500/10',
      borderColor: 'border-amber-500/30',
      textColor: 'text-amber-400',
      empty: 'No moderate impact factors'
    },
    { 
      key: 'protective', 
      title: 'Protective Factors', 
      icon: Shield, 
      color: 'from-emerald-500 to-green-600',
      bgColor: 'bg-emerald-500/10',
      borderColor: 'border-emerald-500/30',
      textColor: 'text-emerald-400',
      empty: 'No protective factors identified'
    },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <Card className="bg-slate-800/50 backdrop-blur-xl border border-white/10">
        <CardHeader className="border-b border-white/10">
          <CardTitle className="flex items-center gap-3 text-white">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-white" />
            </div>
            Risk Factor Analysis
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <div className="grid lg:grid-cols-3 gap-4">
            {sections.map((section) => (
              <div
                key={section.key}
                className={`rounded-xl p-4 border ${section.bgColor} ${section.borderColor}`}
              >
                <div className="flex items-center gap-2 mb-4">
                  <div className={`w-8 h-8 rounded-lg bg-gradient-to-br ${section.color} flex items-center justify-center`}>
                    <section.icon className="w-4 h-4 text-white" />
                  </div>
                  <h3 className={`font-semibold ${section.textColor}`}>{section.title}</h3>
                </div>

                <div className="space-y-3">
                  {factors[section.key].length > 0 ? (
                    factors[section.key].map((factor, index) => (
                      <motion.div
                        key={factor.name}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="bg-slate-800/50 rounded-lg p-3"
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm font-medium text-white">{factor.name}</span>
                          <Badge variant="outline" className={`${section.textColor} border-current text-xs`}>
                            {factor.impact}
                          </Badge>
                        </div>
                        <span className="text-xs text-slate-400">{factor.value}</span>
                      </motion.div>
                    ))
                  ) : (
                    <p className="text-sm text-slate-500 text-center py-4">{section.empty}</p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}