"""Bind the frozen recovery worker to the authorized node scheduling context."""
import common

if __name__ == '__main__':
    wrapper = common.load_module('four_conversation_worker', common.parent.CODE / 'worker.py')
    raise SystemExit(wrapper.implementation().main())
