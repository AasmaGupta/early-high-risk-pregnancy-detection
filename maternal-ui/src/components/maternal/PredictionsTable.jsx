import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { History, User, Download, Loader2, CheckCircle2 } from 'lucide-react';
import { motion } from 'framer-motion';

const mockPredictions = [
  { id: 'P001', date: '2025-02-07', age: 28, systolic: 125, diastolic: 82, sugar: 95, temp: 98.4, hr: 78, risk: 'low', confidence: 92 },
  { id: 'P002', date: '2025-02-07', age: 35, systolic: 145, diastolic: 95, sugar: 140, temp: 98.6, hr: 88, risk: 'mid', confidence: 85 },
  { id: 'P003', date: '2025-02-06', age: 42, systolic: 160, diastolic: 105, sugar: 180, temp: 99.1, hr: 95, risk: 'high', confidence: 94 },
  { id: 'P004', date: '2025-02-06', age: 25, systolic: 118, diastolic: 75, sugar: 88, temp: 98.2, hr: 72, risk: 'low', confidence: 96 },
  { id: 'P005', date: '2025-02-05', age: 38, systolic: 138, diastolic: 88, sugar: 125, temp: 98.8, hr: 82, risk: 'mid', confidence: 78 },
];

const riskBadgeStyles = {
  low: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  mid: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  high: 'bg-rose-500/20 text-rose-400 border-rose-500/30',
};

export default function PredictionsTable() {
  const [downloadingId, setDownloadingId] = useState(null);
  const [downloadedId, setDownloadedId] = useState(null);

  const handleDownload = async (patient) => {
    setDownloadingId(patient.id);
    
    // Simulate download delay
    await new Promise(resolve => setTimeout(resolve, 1000));

    const reportContent = `
PATIENT HISTORY REPORT
======================
Patient ID: ${patient.id}
Generated: ${new Date().toLocaleString()}

CLINICAL PARAMETERS
-------------------
Date of Assessment: ${patient.date}
Age: ${patient.age} years
Systolic BP: ${patient.systolic} mmHg
Diastolic BP: ${patient.diastolic} mmHg
Blood Sugar: ${patient.sugar} mg/dL
Body Temperature: ${patient.temp} °F
Heart Rate: ${patient.hr} bpm

RISK ASSESSMENT
---------------
Risk Level: ${patient.risk.toUpperCase()}
Model Confidence: ${patient.confidence}%

CLINICAL NOTES
--------------
${patient.risk === 'high' 
  ? 'HIGH RISK - Immediate clinical attention recommended. Consider specialist referral.'
  : patient.risk === 'mid'
  ? 'MODERATE RISK - Enhanced monitoring recommended. Schedule follow-up within 2 weeks.'
  : 'LOW RISK - Continue routine prenatal care. Schedule standard follow-up.'}

======================
End of Report
    `;

    const blob = new Blob([reportContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `patient_${patient.id}_history_${patient.date}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    setDownloadingId(null);
    setDownloadedId(patient.id);
    setTimeout(() => setDownloadedId(null), 2000);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <Card className="bg-slate-800/50 backdrop-blur-xl border border-white/10">
        <CardHeader className="border-b border-white/10">
          <CardTitle className="flex items-center gap-3 text-white">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-slate-500 to-gray-500 flex items-center justify-center">
              <History className="w-5 h-5 text-white" />
            </div>
            Historical Predictions
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow className="bg-slate-700/30 border-slate-600/30">
                  <TableHead className="text-slate-300">Patient ID</TableHead>
                  <TableHead className="text-slate-300">Date</TableHead>
                  <TableHead className="text-slate-300">Age</TableHead>
                  <TableHead className="text-slate-300">BP (S/D)</TableHead>
                  <TableHead className="text-slate-300">Sugar</TableHead>
                  <TableHead className="text-slate-300">Temp</TableHead>
                  <TableHead className="text-slate-300">HR</TableHead>
                  <TableHead className="text-slate-300">Risk</TableHead>
                  <TableHead className="text-slate-300">Conf.</TableHead>
                  <TableHead className="text-slate-300">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {mockPredictions.map((pred) => (
                  <TableRow key={pred.id} className="border-slate-600/30 hover:bg-slate-700/30">
                    <TableCell className="font-medium text-white">
                      <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-lg bg-slate-700 flex items-center justify-center">
                          <User className="w-4 h-4 text-slate-400" />
                        </div>
                        {pred.id}
                      </div>
                    </TableCell>
                    <TableCell className="text-slate-400">{pred.date}</TableCell>
                    <TableCell className="text-slate-300">{pred.age}</TableCell>
                    <TableCell className="text-slate-300">{pred.systolic}/{pred.diastolic}</TableCell>
                    <TableCell className="text-slate-300">{pred.sugar}</TableCell>
                    <TableCell className="text-slate-300">{pred.temp}°F</TableCell>
                    <TableCell className="text-slate-300">{pred.hr}</TableCell>
                    <TableCell>
                      <Badge className={riskBadgeStyles[pred.risk]}>
                        {pred.risk.charAt(0).toUpperCase() + pred.risk.slice(1)}
                      </Badge>
                    </TableCell>
                    <TableCell className="font-mono text-teal-400">{pred.confidence}%</TableCell>
                    <TableCell>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleDownload(pred)}
                        disabled={downloadingId === pred.id}
                        className={`${
                          downloadedId === pred.id 
                            ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-400' 
                            : 'bg-slate-700/50 border-slate-600 text-slate-300 hover:bg-slate-600/50 hover:text-white'
                        }`}
                      >
                        {downloadingId === pred.id ? (
                          <Loader2 className="w-4 h-4 animate-spin" />
                        ) : downloadedId === pred.id ? (
                          <CheckCircle2 className="w-4 h-4" />
                        ) : (
                          <Download className="w-4 h-4" />
                        )}
                      </Button>
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