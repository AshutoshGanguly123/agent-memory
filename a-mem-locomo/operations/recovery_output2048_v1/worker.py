"""Use the frozen worker/backend with recovery paths, config and a scope guard."""
import sys
import common


def implementation():
    # common is already bound to this adapter; backend remains the frozen v1 file.
    sys.path.insert(1, str(common.legacy.CODE))
    worker = common.load_module('amem_recovery_worker_impl', common.legacy.CODE / 'worker.py')
    original_run = worker.run_conversation

    def scoped_run(sample, backend, embedder, prompts, config, out, update):
        common.require_target(sample['sample_id'])
        if config != common.CONFIG or out != common.OUT:
            raise RuntimeError('Recovery worker attempted to use another config/output')
        return original_run(sample, backend, embedder, prompts, config, out, update)

    worker.run_conversation = scoped_run
    return worker


if __name__ == '__main__':
    raise SystemExit(implementation().main())
