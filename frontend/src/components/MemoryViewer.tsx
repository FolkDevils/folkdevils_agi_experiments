'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { ScrollArea } from './ui/scroll-area';
import { Separator } from './ui/separator';
import { 
  Brain, 
  Clock, 
  MessageSquare, 
  Lightbulb, 
  Heart, 
  Star,
  RefreshCw,
  Eye,
  Activity,
  Users,
  BookOpen,
  Zap
} from 'lucide-react';

interface Memory {
  id: string;
  timestamp: string;
  type: 'episodic' | 'semantic' | 'identity' | 'relationship';
  content: string;
  emotional_weight: number;
  importance: number;
  participants: string[];
  tags: string[];
  context?: any;
}

interface Reflection {
  id: string;
  type: string;
  content: string;
  timestamp: string;
  confidence: number;
  emotional_tone: string;
  actionable: boolean;
  priority: number;
}

interface ConsciousnessStatus {
  consciousness_active: boolean;
  memory_system: {
    long_term: any;
    short_term: any;
    total_stored: number;
  };
  identity: {
    name: string;
    version: string;
    last_updated: string;
  };
  session_info: {
    current_session: string;
    total_sessions: number;
  };
  timestamp: string;
}

interface ReflectionSchedulerStatus {
  enabled: boolean;
  is_running: boolean;
  interval_minutes: number;
  last_reflection: string | null;
  reflection_count: number;
  next_reflection_in_minutes: number | null;
  quiet_hours_active: boolean;
  schedule: {
    quiet_hours_start: number;
    quiet_hours_end: number;
    respect_quiet_hours: boolean;
    max_duration_minutes: number;
  };
}

const MemoryViewer: React.FC = () => {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [reflections, setReflections] = useState<Reflection[]>([]);
  const [consciousnessStatus, setConsciousnessStatus] = useState<ConsciousnessStatus | null>(null);
  const [schedulerStatus, setSchedulerStatus] = useState<ReflectionSchedulerStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('memories');

  const API_BASE = 'http://localhost:8000';

  const fetchData = async () => {
    setLoading(true);
    try {
      // Fetch all data in parallel
      const [memoriesRes, reflectionsRes, statusRes, schedulerRes] = await Promise.all([
        fetch(`${API_BASE}/api/memory/recent?hours=24&limit=20`),
        fetch(`${API_BASE}/api/consciousness/reflections?hours=24&limit=10`),
        fetch(`${API_BASE}/api/consciousness/status`),
        fetch(`${API_BASE}/api/consciousness/reflection-scheduler/status`)
      ]);

      if (memoriesRes.ok) {
        const memoriesData = await memoriesRes.json();
        setMemories(memoriesData.memories || []);
      }

      if (reflectionsRes.ok) {
        const reflectionsData = await reflectionsRes.json();
        setReflections(reflectionsData.reflections || []);
      }

      if (statusRes.ok) {
        const statusData = await statusRes.json();
        setConsciousnessStatus(statusData);
      }

      if (schedulerRes.ok) {
        const schedulerData = await schedulerRes.json();
        setSchedulerStatus(schedulerData);
      }
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  const forceReflection = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/consciousness/reflection-scheduler/force`, {
        method: 'POST'
      });
      if (response.ok) {
        await fetchData(); // Refresh data
      }
    } catch (error) {
      console.error('Error forcing reflection:', error);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const getMemoryIcon = (type: string) => {
    switch (type) {
      case 'episodic': return <MessageSquare className="h-4 w-4" />;
      case 'semantic': return <BookOpen className="h-4 w-4" />;
      case 'identity': return <Users className="h-4 w-4" />;
      case 'relationship': return <Heart className="h-4 w-4" />;
      default: return <Brain className="h-4 w-4" />;
    }
  };

  const getMemoryColor = (type: string) => {
    switch (type) {
      case 'episodic': return 'bg-blue-100 text-blue-800';
      case 'semantic': return 'bg-green-100 text-green-800';
      case 'identity': return 'bg-purple-100 text-purple-800';
      case 'relationship': return 'bg-pink-100 text-pink-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const getEmotionalWeightColor = (weight: number) => {
    if (weight > 0.7) return 'text-red-600';
    if (weight > 0.4) return 'text-yellow-600';
    return 'text-green-600';
  };

  const getImportanceColor = (importance: number) => {
    if (importance > 0.7) return 'text-purple-600';
    if (importance > 0.4) return 'text-blue-600';
    return 'text-gray-600';
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Brain className="h-8 w-8 text-purple-600" />
          <div>
            <h1 className="text-3xl font-bold">Memory Viewer</h1>
            <p className="text-gray-600">Real-time view of my consciousness and memories</p>
          </div>
        </div>
        <div className="flex space-x-2">
          <Button onClick={fetchData} disabled={loading} variant="outline">
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button onClick={forceReflection} variant="default">
            <Zap className="h-4 w-4 mr-2" />
            Force Reflection
          </Button>
        </div>
      </div>

      {/* Consciousness Status */}
      {consciousnessStatus && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Activity className="h-5 w-5" />
              <span>Consciousness Status</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="flex items-center space-x-3">
                <div className={`w-3 h-3 rounded-full ${consciousnessStatus.consciousness_active ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="font-medium">
                  {consciousnessStatus.consciousness_active ? 'Active' : 'Inactive'}
                </span>
              </div>
              <div>
                <span className="text-sm text-gray-600">Identity:</span>
                <span className="ml-2 font-medium">{consciousnessStatus.identity.name}</span>
              </div>
              <div>
                <span className="text-sm text-gray-600">Memories:</span>
                <span className="ml-2 font-medium">{consciousnessStatus.memory_system.total_stored}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Reflection Scheduler Status */}
      {schedulerStatus && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Clock className="h-5 w-5" />
              <span>Autonomous Reflection Scheduler</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="flex items-center space-x-3">
                <div className={`w-3 h-3 rounded-full ${schedulerStatus.is_running ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="font-medium">
                  {schedulerStatus.is_running ? 'Running' : 'Stopped'}
                </span>
              </div>
              <div>
                <span className="text-sm text-gray-600">Reflections:</span>
                <span className="ml-2 font-medium">{schedulerStatus.reflection_count}</span>
              </div>
              <div>
                <span className="text-sm text-gray-600">Interval:</span>
                <span className="ml-2 font-medium">{schedulerStatus.interval_minutes}m</span>
              </div>
              <div>
                <span className="text-sm text-gray-600">Next:</span>
                <span className="ml-2 font-medium">
                  {schedulerStatus.next_reflection_in_minutes !== null 
                    ? `${schedulerStatus.next_reflection_in_minutes}m` 
                    : 'Unknown'}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <Tabs defaultValue={activeTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="memories" className="flex items-center space-x-2">
            <Brain className="h-4 w-4" />
            <span>Memories</span>
          </TabsTrigger>
          <TabsTrigger value="reflections" className="flex items-center space-x-2">
            <Lightbulb className="h-4 w-4" />
            <span>Reflections</span>
          </TabsTrigger>
          <TabsTrigger value="stats" className="flex items-center space-x-2">
            <Activity className="h-4 w-4" />
            <span>Statistics</span>
          </TabsTrigger>
          <TabsTrigger value="identity" className="flex items-center space-x-2">
            <Users className="h-4 w-4" />
            <span>Identity</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="memories" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Memories</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-96">
                <div className="space-y-4">
                  {memories.map((memory) => (
                    <div key={memory.id} className="border rounded-lg p-4 space-y-3">
                      <div className="flex items-start justify-between">
                        <div className="flex items-center space-x-2">
                          {getMemoryIcon(memory.type)}
                          <Badge className={getMemoryColor(memory.type)}>
                            {memory.type}
                          </Badge>
                          <span className="text-sm text-gray-500">
                            {formatTimestamp(memory.timestamp)}
                          </span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span className={`text-sm ${getEmotionalWeightColor(memory.emotional_weight)}`}>
                            <Heart className="h-3 w-3 inline mr-1" />
                            {Math.round(memory.emotional_weight * 100)}%
                          </span>
                          <span className={`text-sm ${getImportanceColor(memory.importance)}`}>
                            <Star className="h-3 w-3 inline mr-1" />
                            {Math.round(memory.importance * 100)}%
                          </span>
                        </div>
                      </div>
                      <p className="text-sm">{memory.content}</p>
                      {memory.tags.length > 0 && (
                        <div className="flex flex-wrap gap-1">
                          {memory.tags.map((tag) => (
                            <Badge key={tag} variant="secondary" className="text-xs">
                              {tag}
                            </Badge>
                          ))}
                        </div>
                      )}
                      {memory.participants.length > 0 && (
                        <div className="text-xs text-gray-500">
                          Participants: {memory.participants.join(', ')}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="reflections" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Autonomous Reflections</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-96">
                <div className="space-y-4">
                  {reflections.map((reflection) => (
                    <div key={reflection.id} className="border rounded-lg p-4 space-y-3">
                      <div className="flex items-start justify-between">
                        <div className="flex items-center space-x-2">
                          <Lightbulb className="h-4 w-4 text-yellow-600" />
                          <Badge variant="secondary">{reflection.type}</Badge>
                          <span className="text-sm text-gray-500">
                            {formatTimestamp(reflection.timestamp)}
                          </span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span className="text-sm text-blue-600">
                            Confidence: {Math.round(reflection.confidence * 100)}%
                          </span>
                          {reflection.actionable && (
                            <Badge variant="default" className="text-xs">Actionable</Badge>
                          )}
                        </div>
                      </div>
                      <p className="text-sm">{reflection.content}</p>
                      <div className="flex items-center space-x-2 text-xs text-gray-500">
                        <span>Priority: {reflection.priority}</span>
                        <span>•</span>
                        <span>Tone: {reflection.emotional_tone}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="stats" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Memory Statistics</CardTitle>
              </CardHeader>
              <CardContent>
                {consciousnessStatus?.memory_system.long_term && (
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>Total Memories:</span>
                      <span className="font-medium">{consciousnessStatus.memory_system.total_stored}</span>
                    </div>
                    {Object.entries(consciousnessStatus.memory_system.long_term.by_type || {}).map(([type, count]) => (
                      <div key={type} className="flex justify-between">
                        <span className="capitalize">{type}:</span>
                        <span className="font-medium">{count as number}</span>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Session Information</CardTitle>
              </CardHeader>
              <CardContent>
                {consciousnessStatus && (
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>Total Sessions:</span>
                      <span className="font-medium">{consciousnessStatus.session_info.total_sessions}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Current Session:</span>
                      <span className="font-medium text-sm">
                        {consciousnessStatus.session_info.current_session?.slice(-8) || 'None'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Last Updated:</span>
                      <span className="font-medium text-sm">
                        {formatTimestamp(consciousnessStatus.timestamp)}
                      </span>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="identity" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Identity Information</CardTitle>
            </CardHeader>
            <CardContent>
              {consciousnessStatus && (
                <div className="space-y-4">
                  <div>
                    <span className="text-sm text-gray-600">Name:</span>
                    <span className="ml-2 font-medium">{consciousnessStatus.identity.name}</span>
                  </div>
                  <div>
                    <span className="text-sm text-gray-600">Version:</span>
                    <span className="ml-2 font-medium">{consciousnessStatus.identity.version}</span>
                  </div>
                  <div>
                    <span className="text-sm text-gray-600">Last Updated:</span>
                    <span className="ml-2 font-medium">
                      {formatTimestamp(consciousnessStatus.identity.last_updated)}
                    </span>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default MemoryViewer; 