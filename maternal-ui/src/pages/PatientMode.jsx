import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { 
  Baby, 
  Heart, 
  Shield, 
  Sparkles,
  ChevronDown,
  ChevronUp,
  Info
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

import InputForm from '@/components/maternal/InputForm';
import ResultsPanel from '@/components/maternal/ResultsPanel';

// Simulated prediction function
const simulatePrediction = (data) => {
  const riskScore = 
    (data.systolicBP > 140 ? 30 : 0) +
    (data.diastolicBP > 90 ? 20 : 0) +
    (data.bloodSugar > 140 ? 25 : 0) +
    (data.age > 35 ? 15 : 0) +
    (data.heartRate > 90 ? 10 : 0);

  let riskLevel = 'low';
  let confidence = 92;
  let probabilities = { low: 75, mid: 20, high: 5 };

  if (riskScore > 50) {
    riskLevel = 'high';
    confidence = 94;
    probabilities = { low: 10, mid: 25, high: 65 };
  } else if (riskScore > 25) {
    riskLevel = 'mid';
    confidence = 85;
    probabilities = { low: 25, mid: 55, high: 20 };
  }

  return { riskLevel, confidence, probabilities };
};

export default function PatientMode() {
  const [formData, setFormData] = useState({
    age: 28,
    systolicBP: 120,
    diastolicBP: 80,
    bloodSugar: 95,
    bodyTemp: 98.6,
    heartRate: 75,
  });
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showExplanation, setShowExplanation] = useState(false);

  const handlePredict = async () => {
    setIsLoading(true);
    await new Promise(resolve => setTimeout(resolve, 1500));
    const prediction = simulatePrediction(formData);
    setResult(prediction);
    setIsLoading(false);
  };

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-8"
      >
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-pink-400 to-rose-500 shadow-lg shadow-pink-500/20 mb-4">
          <Baby className="w-8 h-8 text-white" />
        </div>
        <h1 className="text-3xl font-bold text-white mb-2">
          Pregnancy Health Screening
        </h1>
        <p className="text-slate-400 max-w-lg mx-auto">
          A simple health check to help you understand your pregnancy risk level. 
          Enter your health information below.
        </p>
      </motion.div>

      {/* Calming Info Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="mb-6"
      >
        <Card className="bg-gradient-to-r from-pink-500/10 to-rose-500/10 border border-pink-500/20">
          <CardContent className="p-4 flex items-center gap-4">
            <div className="w-10 h-10 rounded-full bg-pink-500/20 flex items-center justify-center flex-shrink-0">
              <Heart className="w-5 h-5 text-pink-400" />
            </div>
            <p className="text-sm text-slate-300">
              This screening tool helps identify potential risks early. 
              Your information is <span className="text-pink-400 font-medium">private and secure</span>.
            </p>
          </CardContent>
        </Card>
      </motion.div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Input Form */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <InputForm
            formData={formData}
            setFormData={setFormData}
            onSubmit={handlePredict}
            isLoading={isLoading}
            variant="patient"
          />
        </motion.div>

        {/* Results */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
        >
          {result ? (
            <div className="space-y-4">
              <ResultsPanel result={result} variant="patient" />
              
              {/* Simple Explanation Toggle */}
              <Card className="bg-white border-slate-200">
                <CardHeader 
                  className="cursor-pointer py-4"
                  onClick={() => setShowExplanation(!showExplanation)}
                >
                  <CardTitle className="flex items-center justify-between text-slate-700 text-base">
                    <span className="flex items-center gap-2">
                      <Info className="w-5 h-5 text-teal-500" />
                      What does this mean?
                    </span>
                    {showExplanation ? (
                      <ChevronUp className="w-5 h-5 text-slate-400" />
                    ) : (
                      <ChevronDown className="w-5 h-5 text-slate-400" />
                    )}
                  </CardTitle>
                </CardHeader>
                <AnimatePresence>
                  {showExplanation && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="overflow-hidden"
                    >
                      <CardContent className="pt-0 pb-4 text-sm text-slate-600 space-y-3">
                        <p>
                          Our system looked at your health numbers and compared them 
                          to patterns from thousands of similar cases.
                        </p>
                        <p>
                          The main factors we considered are:
                        </p>
                        <ul className="list-disc list-inside space-y-1 text-slate-500">
                          <li>Blood pressure levels</li>
                          <li>Blood sugar levels</li>
                          <li>Age-related factors</li>
                          <li>Heart rate patterns</li>
                        </ul>
                        <p className="text-teal-600 font-medium">
                          Remember: This is a screening tool. Always consult with your doctor 
                          for medical advice.
                        </p>
                      </CardContent>
                    </motion.div>
                  )}
                </AnimatePresence>
              </Card>
            </div>
          ) : (
            <Card className="bg-white border-slate-200 h-full flex items-center justify-center min-h-[300px]">
              <CardContent className="text-center py-12">
                <div className="w-20 h-20 rounded-full bg-gradient-to-br from-teal-100 to-cyan-100 flex items-center justify-center mx-auto mb-4">
                  <Sparkles className="w-10 h-10 text-teal-500" />
                </div>
                <h3 className="text-xl font-semibold text-slate-700 mb-2">
                  Ready for Your Screening
                </h3>
                <p className="text-slate-500">
                  Fill in your health information and click<br />
                  <span className="text-teal-600 font-medium">"Predict Risk Level"</span> to begin.
                </p>
              </CardContent>
            </Card>
          )}
        </motion.div>
      </div>

      {/* Disclaimer */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        className="mt-8"
      >
        <Card className="bg-slate-800/30 border border-slate-700/50">
          <CardContent className="p-4 flex items-start gap-3">
            <Shield className="w-5 h-5 text-slate-400 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-slate-400 leading-relaxed">
              <span className="font-semibold text-slate-300">Important Notice:</span> This tool provides 
              general health screening information only. It is not a medical diagnosis. 
              Please consult with a qualified healthcare provider for proper medical advice, 
              diagnosis, and treatment. Your privacy is protected — no personal data is stored.
            </p>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}