import serial
import time

"""
Install Package
- PySerial(pip install pyserial)

Setup
- Connect USB port of Scale(LED red, yellow(blinking))
- Check USB port name(ex: /dev/ttyUSB or COM2)
- Modify port name at _main()_
- Power on scale
- Set Format to 5, unit to mg

My Setup
- Check USB port name(ex: /dev/ttyUSB or COM2)
- Set Format to 5
- Set unit to mg
- sudo chmod a+rw /dev/ttyUSB0

Variable:
	self._stable: S means stable, U means unstable states
	self._value: amount  (If value is 2000000.0, it means Over Load)
	self._unit:  unit of amount
	self._count: continious failed count

"""


class ScaleITX120Interface(object):
    def __init__(self, port="/dev/ttyUSB0"):
        self._stable = 'U'
        self._value = 0.0
        self._unit = 'none'
        self._count = 0

        try:
            # Serial Port Use Format5
            self._port = serial.Serial(port)
            # import ipdb
            # ipdb.set_trace()
            self._port.close()
            self._port.baudrate = 2400
            self._port.bytesize = serial.SEVENBITS
            self._port.stopbits = serial.STOPBITS_TWO
            self._port.parity = serial.PARITY_EVEN
            self._port.timeout = 0.2
            self._port.open()
            self._portClose = True
            self._port.write(b'D09\r\n')
            time.sleep(0.2)
            self._port.reset_input_buffer()
            self._port.reset_output_buffer()
            self._port.read_all()
        except:
            print('Port Open Error')

    def __del__(self):
        if not self._portClose:
            self._port.flush()
            self._port.close()
            self._portClose = True

    def readPackets(self):
        try:
            self._port.write(b'D03\r\n')
            time.sleep(0.1)
            while True:
                buf = self._port.readline()
                # TODO change try parse
                if (len(buf) == 17):
                    self._stable = buf[0: 1].decode('ascii')
                    self._unit = buf[13: 15].decode('ascii')  # Expected mg
                    if (self._unit == 'mg'):
                        self._value = float(buf[3:12].decode('ascii'))
                        self._count = 0
                    elif (buf[8:10].decode('ascii') == 'OL'):
                        self._value = 2000000.0
                    else:
                        print(buf[8:11].decode('ascii'))
                        self._count = self._count + 1
                    print("%s, %F[%s]%d" % (self._stable, self._value, self._unit, self._count))
        except KeyboardInterrupt:
            print('Keyboard Interrupt')
            self._port.write(b'D09\r\n')
            return

    def readValue(self):
        print('wait 10 second for scaling')
        try:
            self._port.write(b'D03\r\n')
            time.sleep(10)  # wait for stable scaling
            while True:
                self._port.reset_input_buffer()
                buf = self._port.readline()
                # TODO change try parse
                if (len(buf) == 17):
                    self._stable = buf[0: 1].decode('ascii')
                    self._unit = buf[13: 15].decode('ascii')  # Expected mg
                    if (self._unit == 'mg'):
                        self._value = float(buf[3:12].decode('ascii'))
                        self._count = 0
                    elif (buf[8:10].decode('ascii') == 'OL'):
                        self._value = 2000000.0
                    else:
                        print(buf[8:11].decode('ascii'))
                        # self._count = self._count + 1
                    print("%s, %F[%s]%d" % (self._stable, self._value, self._unit, self._count))
                    if self._stable:
                        print("powder: {}[mg]".format(self._value))
                        return self._value
        except KeyboardInterrupt:
            print('Keyboard Interrupt')
            self._port.write(b'D09\r\n')
            return

    def setZero(self):
        self._port.write(b'D09\r\n')
        time.sleep(0.2)
        self._port.read_all()
        self._port.write(b'TARE\r\n')
        time.sleep(0.2)


def test1():
    print('start')
    scale = ScaleITX120Interface(port="/dev/ttyUSB0")
    scale.readPackets()
    # scale.setZero()


def test2():
    scale = ScaleITX120Interface(port="/dev/ttyUSB0")
    scale.readValue()


if __name__ == '__main__':
    test2()
