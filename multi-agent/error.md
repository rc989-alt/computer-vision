# Task execution tracker
class TaskExecutionTracker:
    def __init__(self):
        self.task_results = []
        self.start_time = datetime.now()
        self.current_task = None

    def start_task(self, task_id, action, priority):
        self.current_task = {
            'task_id': task_id,
            'action': action,
            'priority': priority,
            'status': 'in_progress',
            'start_time': datetime.now().isoformat(),
            'outputs': [],
            'errors': [],
            'agent_responses': {}
        }
        print(f"\n🚀 Starting Task {task_id}: {action}")
        print(f"   Priority: {priority}")
        print(f"   Started: {self.current_task['start_time']}")

    def log_agent_response(self, agent_name, response):
        if self.current_task:
            self.current_task['agent_responses'][agent_name] = response
            print(f"   ✅ {agent_name} responded ({len(response)} chars)")

    def log_output(self, output_type, content, file_path=None):
        if self.current_task:
            self.current_task['outputs'].append({
                'type': output_type,
                'content': content,
                'file_path': file_path,
                'timestamp': datetime.now().isoformat()
            })
            print(f"   📄 Output: {output_type}" + (f" → {file_path}" if file_path else ""))

    def log_error(self, error_msg):
        if self.current_task:
            self.current_task['errors'].append({
                'message': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            print(f"   ❌ Error: {error_msg}")

    def complete_task(self, status='completed'):
        if self.current_task:
            self.current_task['status'] = status
            self.current_task['end_time'] = datetime.now().isoformat()
            self.task_results.append(self.current_task)

            duration = (datetime.fromisoformat(self.current_task['end_time']) -
                       datetime.fromisoformat(self.current_task['start_time'])).total_seconds()

            print(f"   ✅ Task completed in {duration:.1f}s")
            print(f"   Status: {status}")
            print(f"   Outputs: {len(self.current_task['outputs'])}")
            print(f"   Errors: {len(self.current_task['errors'])}")

            self.current_task = None

    def get_summary(self):
        completed = len([t for t in self.task_results if t['status'] == 'completed'])
        failed = len([t for t in self.task_results if t['status'] == 'failed'])
        total_duration = (datetime.now() - self.start_time).total_seconds()

        return {
            'total_tasks': len(self.task_results),
            'completed': completed,
            'failed': failed,
            'total_duration_seconds': total_duration,
            'task_results': self.task_results
        }

tracker = TaskExecutionTracker()
print("✅ Task execution tracker initialized")