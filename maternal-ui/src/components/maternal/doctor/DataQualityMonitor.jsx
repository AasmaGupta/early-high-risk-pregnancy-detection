import React, { useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  ShieldCheck, 
  AlertTriangle, 
  AlertCircle,
  CheckCircle2,
  XCircle
} from 'lucide-react';
import { motion } from 'framer-motion';

const normalRanges = {
  age: { min: 15, max: 50, unit: 'years', label: 'Age' },
  systolicBP: { min: 80, max: 180, unit: 'mmHg', label: 'Systolic BP' },
  diastolicBP: { min: 50, max: 120, unit: 'mmHg', label: 'Diastolic BP' },
  bloodSugar: { min: 60, max: 250, unit: 'mg/dL', label: 'Blood Sugar' },
  bodyTemp: { min: 96, max: 102, unit: '°F', label: 'Body Temp' },
  heartRate: { min: 50, max: 110, unit: 'bpm', label: 'Heart Rate' },
};

const validateData = (formData) => {
  const issues = [];
  const warnings = [];
  const valid = [];

  if (!formData) {
    return { issues: [{ field: 'All', message: 'No data provided' }], warnings: [], valid: [], score: 0 };
  }

  Object.entries(normalRanges).forEach(([key, range]) => {
    const value = formData[key];
    
    if (value === undefined || value === null || value === '') {
      issues.push({ field: range.label, message: 'Missing value', type: 'missing' });
    } else if (value < range.min || value > range.max) {
      issues.push({ 
        field: range.label, 
        message: `Value ${value} ${range.unit} is outside expected range (${range.min}-${range.max})`,
        type: 'out_of_range',
        value
      });
    } else {
      // Check for warning thresholds
      const rangeSpan = range.max - range.min;
      const lowerWarning = range.min + rangeSpan * 0.1;
      const upperWarning = range.max - rangeSpan * 0.1;
      
      if (value < lowerWarning || value > upperWarning) {
        warnings.push({
          field: range.label,
          message: `Value ${value} ${range.unit} is near boundary`,
          value
        });
      } else {
        valid.push({ field: range.label, value, unit: range.unit });
      }
    }
  });

  // BP consistency check
  if (formData.systolicBP && formData.diastolicBP) {
    if (formData.diastolicBP >= formData.systolicBP) {
      issues.push({
        field: 'Blood Pressure',
        message: 'Diastolic BP cannot be greater than or equal to Systolic BP',
        type: 'anomaly'
      });
    }
  }

  const totalFields = Object.keys(normalRanges).length;
  const validFields = valid.length + warnings.length;
  const score = Math.round((validFields / totalFields) * 100);

  return { issues, warnings, valid, score };
};

export default function DataQualityMonitor({ formData }) {
  const validation = useMemo(() => validateData(formData), [formData]);
  
  const getScoreColor = (score) => {
    if (score >= 80) return { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400' };
    if (score >= 60) return { bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-400' };
    return { bg: 'bg-rose-500/10', border: 'border-rose-500/30', text: 'text-rose-400' };
  };

  const scoreStyle = getScoreColor(validation.score);
  const hasIssues = validation.issues.length > 0;
  const hasWarnings = validation.warnings.length > 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <Card className="bg-slate-800/50 backdrop-blur-xl border border-white/10">
        <CardHeader className="border-b border-white/10">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-3 text-white">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-sky-500 to-blue-500 flex items-center justify-center">
                <ShieldCheck className="w-5 h-5 text-white" />
              </div>
              Data Quality Monitor
            </CardTitle>
            <Badge className={`${scoreStyle.bg} ${scoreStyle.text} border ${scoreStyle.border}`}>
              Quality Score: {validation.score}%
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="p-6">
          {/* Quality Summary */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className={`rounded-xl p-4 text-center ${validation.issues.length === 0 ? 'bg-emerald-500/10 border border-emerald-500/30' : 'bg-rose-500/10 border border-rose-500/30'}`}>
              {validation.issues.length === 0 ? (
                <CheckCircle2 className="w-6 h-6 text-emerald-400 mx-auto mb-2" />
              ) : (
                <XCircle className="w-6 h-6 text-rose-400 mx-auto mb-2" />
              )}
              <p className={`text-2xl font-bold ${validation.issues.length === 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {validation.issues.length}
              </p>
              <p className="text-xs text-slate-400">Critical Issues</p>
            </div>
            <div className={`rounded-xl p-4 text-center ${validation.warnings.length === 0 ? 'bg-emerald-500/10 border border-emerald-500/30' : 'bg-amber-500/10 border border-amber-500/30'}`}>
              <AlertTriangle className={`w-6 h-6 mx-auto mb-2 ${validation.warnings.length === 0 ? 'text-emerald-400' : 'text-amber-400'}`} />
              <p className={`text-2xl font-bold ${validation.warnings.length === 0 ? 'text-emerald-400' : 'text-amber-400'}`}>
                {validation.warnings.length}
              </p>
              <p className="text-xs text-slate-400">Warnings</p>
            </div>
            <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-4 text-center">
              <CheckCircle2 className="w-6 h-6 text-emerald-400 mx-auto mb-2" />
              <p className="text-2xl font-bold text-emerald-400">{validation.valid.length}</p>
              <p className="text-xs text-slate-400">Valid Fields</p>
            </div>
          </div>

          {/* Issues List */}
          {hasIssues && (
            <div className="mb-4">
              <h4 className="text-sm font-medium text-rose-400 mb-3 flex items-center gap-2">
                <AlertCircle className="w-4 h-4" />
                Critical Issues (Must Fix)
              </h4>
              <div className="space-y-2">
                {validation.issues.map((issue, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="bg-rose-500/10 border border-rose-500/30 rounded-lg p-3 flex items-start gap-3"
                  >
                    <XCircle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm font-medium text-rose-400">{issue.field}</p>
                      <p className="text-xs text-slate-400">{issue.message}</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          )}

          {/* Warnings List */}
          {hasWarnings && (
            <div className="mb-4">
              <h4 className="text-sm font-medium text-amber-400 mb-3 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4" />
                Warnings (Review Recommended)
              </h4>
              <div className="space-y-2">
                {validation.warnings.map((warning, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3 flex items-start gap-3"
                  >
                    <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm font-medium text-amber-400">{warning.field}</p>
                      <p className="text-xs text-slate-400">{warning.message}</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          )}

          {/* All Valid Message */}
          {!hasIssues && !hasWarnings && (
            <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-4 text-center">
              <CheckCircle2 className="w-8 h-8 text-emerald-400 mx-auto mb-2" />
              <p className="text-emerald-400 font-medium">All inputs validated successfully</p>
              <p className="text-xs text-slate-400 mt-1">Data quality meets requirements for prediction</p>
            </div>
          )}

          {/* Prediction Readiness */}
          <div className={`mt-4 p-3 rounded-lg ${hasIssues ? 'bg-rose-500/10 border border-rose-500/30' : 'bg-teal-500/10 border border-teal-500/30'}`}>
            <p className={`text-sm font-medium ${hasIssues ? 'text-rose-400' : 'text-teal-400'}`}>
              {hasIssues 
                ? '⚠️ Please resolve critical issues before running prediction'
                : '✓ Data ready for prediction analysis'
              }
            </p>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}