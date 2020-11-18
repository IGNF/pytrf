"""
    Class for handling dates in various formats
"""

# External imports
#-----------------
import time
import datetime
import calendar

# Data
#-----

# Origin of GPS time scale
gps0  = calendar.timegm(datetime.datetime(1980, 1, 6, 0, 0, 0).timetuple())

# Origin of J2000 time scale
j2000 = calendar.timegm(datetime.datetime(2000, 1, 1, 12, 0, 0).timetuple())
mjd2000 = 51544.5

# date class
#-----------
class date:
  
    """
    Class for handling dates in various formats

    A date instance can be initialized in one of the following ways:
    
        t = date()
        t = date(tsys)
        t = date.from_mjd(mjd)
        t = date.from_ymdhms(year, month, day, [hour, minute, second])
        t = date.from_wd(week, dow)
        t = date.from_tsnx(tsnx)
        t = date.from_tiso(tiso)
        t = date.from_ydec(ydec)
        t = date.from_wdec(wdec)

    See the documentation of each constructor method for details.

    Once initialized, each date instance has the following attributes:

        tsys : System time
        mjd  : Modified Julian Day
        yyyy : 4-char year
        yy   : 2-char year
        mm   : 2-char month
        dd   : 2-char day of month
        hour : 2-char hour
        min  : 2-char minute
        sec  : 2-char second
        doy  : 3-char day of year
        week : 4-char GPS week
        dow  : 1-char day of week
        wk   : 2-char week of year

    Each date instance has the following methods:

        __str__  : Print date in 'YYYY-MM-DD hh:mm:ss' format
        add_s(n) : Add n seconds to instance
        add_h(n) : Add n hours to instance
        add_d(n) : Add n days to instance
        ydec()   : Return decimal year
        wdec()   : Return decimal GPS week
        tsnx()   : Return date in SINEX date format ('yy:ddd:sssss')
        tiso()   : Return date in ISO format ('YYYY-MM-DDThh:mm:ss')
        
    """

    def __init__(self, *args):
      
        """
        Initialize a date instance to "now" or from system time.

        Returns
        -------
        t : date instance

        Parameters
        ----------
        tsys : float, optional
            System time
        """

        # If no argument, t = now
        if (len(args) == 0):
            self.tsys = time.time()

        # If one argument, it should be the system time
        elif (len(args) == 1):
            self.tsys = args[0]

        # Intermediate struct_time instance
        st = time.gmtime(self.tsys)

        # Define attributes of the date instance
        self.mjd = mjd2000 + float(self.tsys - j2000) / 86400
        self.yyyy = time.strftime('%Y', st)
        self.yy = time.strftime('%y', st)
        self.mm = time.strftime('%m', st)
        self.dd = time.strftime('%d', st)
        self.hour = time.strftime('%H', st)
        self.min = time.strftime('%M', st)
        self.sec = time.strftime('%S', st)
        self.doy = time.strftime('%j', st)
        self.dow = time.strftime('%w', st)
        self.week = '{0:04d}'.format(int((self.tsys-gps0) / 86400 / 7))
        self.wk = '{0:02d}'.format(int(float(self.doy) / 7) + 1)
        
    def __str__(self):
      
        """
        Print date in 'YYYY-MM-DD hh:mm:ss' format

        Prints
        ------
        Date in 'YYYY-MM-DD hh:mm:ss' format
        """
        
        s = self.yyyy + '-' + self.mm + '-' + self.dd
        s += ' ' + self.hour + ':' + self.min + ':' + self.sec
        return s
      
    def add_s(self, n):
      
        """
        Add n seconds to date object

        Parameters
        ----------
        n : float
            Number of seconds to add
        """
        
        self.__init__(self.tsys + n)
        
    def add_h(self, n):
      
        """
        Add n hours to date object

        Parameters
        ----------
        n : float
            Number of hours to add
        """
        
        self.__init__(self.tsys + n*3600)
        
    def add_d(self, n):
      
        """
        Add n days to date object

        Parameters
        ----------
        n : float
            Number of days to add
        """
        
        self.__init__(self.tsys + n*86400)
        
    def ydec(self):
      
        """
        Return decimal year

        Returns
        -------
        y : float
            Decimal year
        """
      
        # Get necessary numerical values
        y = int(self.yyyy)
        d = int(self.doy)
        h = int(self.hour)
        m = int(self.min)
        s = int(self.sec)
        
        # Case of a leap year
        if (((y % 4 == 0) and (y % 100 != 0)) or (y % 400 == 0)):
            return y + (d-1 + (h + (m + s / 60) / 60) / 24) / 366
        
        # Case of a 365-day year
        else:
            return y + (d-1 + (h + (m + s / 60) / 60) / 24) / 365
      
    def wdec(self):
      
        """
        Return decimal GPS week

        Returns
        -------
        w : float
            Decimal GPS week
        """
      
        # Get necessary numerical values
        w = int(self.week)
        d = int(self.dow)
        h = int(self.hour)
        m = int(self.min)
        s = int(self.sec)
        
        return w + (d + (h + (m + s / 60) / 60) / 24) / 7
    
    def tsnx(self):
      
        """
        Return date in SINEX format

        Returns
        -------
        s : str
            date in SINEX date format ('yy:ddd:sssss')
        """
      
        # Second of day
        sec = 3600*int(self.hour) + 60*int(self.min) + int(self.sec)
        
        # Date in SINEX format
        return self.yy + ':' + self.doy + ':' + '{0:05d}'.format(sec)
      
    def tiso(self):
      
        """
        Return date in ISO format

        Returns
        -------
        s : str
            date in ISO format ('YYYY-MM-DDThh:mm:ss')
        """
        s = self.yyyy + '-' + self.mm + '-' + self.dd
        s = s + 'T' + self.hour + ':' + self.min + ':' + self.sec
        return s
          
    @classmethod
    def from_mjd(self, mjd):
      
        """
        Initialize a date instance from Modified Julian Day.

        Returns
        -------
        t : date instance

        Parameters
        ----------
        mjd : float
            Modified Julian Day
        """
        
        return date(j2000 + (mjd - mjd2000) * 86400)
    
    @classmethod
    def from_ymdhms(self, y, m, d, hour=0, minute=0, second=0):
      
        """
        Initialize a date instance from year, month, day, [hour, minute, second]

        Returns
        -------
        t : date instance

        Parameters
        ----------
        y : int
            Year
        m : int
            Month
        d : int
            Day in month
        hour : int
            Hour
        minute : int
            Minute
        second : int
            Second
        """
        
        t = datetime.datetime(y, m, d, hour, minute, second)
        return date(calendar.timegm(t.timetuple()))
 
    @classmethod
    def from_wd(self, w, d):
      
        """
        Initialize a date instance from GPS week and day of week

        Returns
        -------
        t : date instance

        Parameters
        ----------
        w : int
            GPS week
        d : float
            Day of week
        """
 
        return date(gps0 + (7*w + d) * 86400)
      
    @classmethod
    def from_tsnx(self, s):
      
        """
        Initialize a date instance from date in SINEX format

        Returns
        -------
        t : date instance

        Parameters
        ----------
        s : str
            Date in SINEX format ('yy:ddd:sssss')
        """
        
        t = calendar.timegm(time.strptime(s[0:6], '%y:%j'))
        return date(t + int(s[7:12]))
      
    @classmethod
    def from_tiso(self, s):
      
        """
        Initialize a date instance from date in ISO format

        Returns
        -------
        t : date instance

        Parameters
        ----------
        s : str
            Date in ISO format ('YYYY-MM-DDThh:mm:ss')
        """
        
        return date.from_ymdhms(int(s[0:4]), int(s[5:7]), int(s[8:10]), int(s[11:13]), int(s[14:16]), int(s[17:19]))

    @classmethod
    def from_ydec(self, y):
      
        """
        Initialize a date instance from decimal year

        Returns
        -------
        t : date instance

        Parameters
        ----------
        y : float
            Decimal year
        """
        
        # Split decimal year into integer and fractional parts
        yint = int(y)
        yfra = y - yint
        
        # Initialize date object on January 1st of year yint
        t = date.from_ymdhms(yint, 1, 1, 0, 0)
        
        # Add appropriate number of days
        # = fractional year multiplier by either 365 or 366
        if (((y % 4 == 0) and (y % 100 != 0)) or (y % 400 == 0)):
            t.add_d(yfra*366)
        else:
            t.add_d(yfra*365)
        
        return t
      
    @classmethod
    def from_wdec(self, w):
      
        """
        Initialize a date instance from decimal GPS week

        Returns
        -------
        t : date instance

        Parameters
        ----------
        w : float
            Decimal GPS week
        """
        
        # Split decimal GPS week into integer and fractional parts
        wint = int(w)
        wfra = w - wint
        
        # Initialize date object from GPS week and day of week
        return date.from_wd(wint, wfra*7)
