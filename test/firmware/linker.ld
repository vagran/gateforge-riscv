__stack_size = 64K;
ENTRY(_resetHandler)

EXTERN(_resetHandler)

SECTIONS
{
	. = 0;

	.text.reset : ALIGN(16) {
		*(.reset)
	}

	. = 0x1000;

	.text : ALIGN(16) {
		*(.text)
	}

    . = ALIGN(4);
    _etext = .;

	.data : ALIGN(16) {
        _data = .;
		*(.rodata)
		*(.rodata.*)
		*(.data)
		*(.data.*)
        . = ALIGN(4);
        _edata = .;
	}

	.bss : ALIGN( 16 ) {
        _bss = .;
		*(.bss)
		*(.bss.*)
        . = ALIGN(4);
        _ebss = .;
	}

	.stack : ALIGN( 16 ) {
		_estack = .;
	}

    _sstack = _estack + __stack_size;
}
